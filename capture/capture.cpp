// capture.cpp — see capture.h. Pure X11/GLX/GL, no glfw. The caller owns the
// Display and GL context.

#include "capture.h"

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/extensions/Xcomposite.h>
#include <X11/extensions/Xfixes.h>
#include <X11/extensions/shape.h>
#include <GL/gl.h>
#include <GL/glx.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

typedef void (*PFNBindTexImageEXT)(Display *, GLXDrawable, int, const int *);
typedef void (*PFNReleaseTexImageEXT)(Display *, GLXDrawable, int);
static PFNBindTexImageEXT    p_bind;
static PFNReleaseTexImageEXT p_release;

#define MAXWINS 128

struct Cap {
    Window    xid;
    Pixmap    pix;
    GLXPixmap glx;
    GLuint    tex;
    int       x, y, w, h, depth;
    bool      seen;
    bool      bound;
};

static Display *dpy        = nullptr;
static Window   root       = 0;
static Window   self_win   = 0;
static Window   cow        = 0;     // the Composite overlay window, while we hold it
static bool     live       = false;
static bool     redirected = false;

static PointerBarrier barriers[4]; static int nbarriers = 0;

static int rx = 0, ry = 0, rw = 0, rh = 0;

static Cap   caps[MAXWINS]; static int ncaps = 0;
static CapWin pub[MAXWINS]; static int npub  = 0;

static GLXFBConfig cands[256]; static int ncand = 0;
struct DCache { int depth; GLXFBConfig fbc; int fmt; bool set; };
static DCache dcache[8]; static int ndc = 0;

static unsigned char g_err = 0;
static int err_handler(Display *, XErrorEvent *e) { g_err = e->error_code; return 0; }

static void gather_candidates()
{
    ncand = 0;
    int n = 0, scr = DefaultScreen(dpy);
    GLXFBConfig *cfgs = glXGetFBConfigs(dpy, scr, &n);
    if (!cfgs) return;
    for (int i = 0; i < n && ncand < 256; i++) {
        int draw = 0, targets = 0, rgb = 0, rgba = 0;
        glXGetFBConfigAttrib(dpy, cfgs[i], GLX_DRAWABLE_TYPE, &draw);
        if (!(draw & GLX_PIXMAP_BIT)) continue;
        glXGetFBConfigAttrib(dpy, cfgs[i], GLX_BIND_TO_TEXTURE_TARGETS_EXT, &targets);
        if (!(targets & GLX_TEXTURE_2D_BIT_EXT)) continue;
        glXGetFBConfigAttrib(dpy, cfgs[i], GLX_BIND_TO_TEXTURE_RGB_EXT,  &rgb);
        glXGetFBConfigAttrib(dpy, cfgs[i], GLX_BIND_TO_TEXTURE_RGBA_EXT, &rgba);
        if (!rgb && !rgba) continue;
        cands[ncand++] = cfgs[i];
    }
    XFree(cfgs);
}

static DCache *dcache_for(int depth)
{
    for (int i = 0; i < ndc; i++) if (dcache[i].depth == depth) return &dcache[i];
    if (ndc < 8) { dcache[ndc] = {depth, nullptr, 0, false}; return &dcache[ndc++]; }
    return nullptr;
}

static GLXPixmap try_create(Pixmap pix, GLXFBConfig fbc, int fmt)
{
    const int attrs[] = { GLX_TEXTURE_TARGET_EXT, GLX_TEXTURE_2D_EXT,
                          GLX_TEXTURE_FORMAT_EXT, fmt, None };
    g_err = 0;
    GLXPixmap gp = glXCreatePixmap(dpy, fbc, pix, attrs);
    XSync(dpy, False);
    if (g_err) return 0;
    return gp;
}

static bool acquire(Cap *c)
{
    if (c->glx) { glXDestroyPixmap(dpy, c->glx); c->glx = 0; }
    if (c->pix) { XFreePixmap(dpy, c->pix);      c->pix = 0; }
    c->bound = false;

    g_err = 0;
    c->pix = XCompositeNameWindowPixmap(dpy, c->xid);
    XSync(dpy, False);
    if (g_err || !c->pix) { c->pix = 0; return false; }

    DCache *dc = dcache_for(c->depth);
    if (dc && dc->set) c->glx = try_create(c->pix, dc->fbc, dc->fmt);
    if (!c->glx) {
        int fmts[2] = { GLX_TEXTURE_FORMAT_RGB_EXT, GLX_TEXTURE_FORMAT_RGBA_EXT };
        for (int ci = 0; ci < ncand && !c->glx; ci++)
            for (int fi = 0; fi < 2 && !c->glx; fi++) {
                GLXPixmap gp = try_create(c->pix, cands[ci], fmts[fi]);
                if (gp) { c->glx = gp; if (dc) { dc->fbc = cands[ci]; dc->fmt = fmts[fi]; dc->set = true; } }
            }
    }
    if (!c->glx) return false;

    if (!c->tex) {
        glGenTextures(1, &c->tex);
        glBindTexture(GL_TEXTURE_2D, c->tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
    return true;
}

static Cap *find_cap(Window w) { for (int i = 0; i < ncaps; i++) if (caps[i].xid == w) return &caps[i]; return nullptr; }

static void drop_cap(int idx)
{
    Cap *c = &caps[idx];
    if (c->glx) glXDestroyPixmap(dpy, c->glx);
    if (c->pix) XFreePixmap(dpy, c->pix);
    if (c->tex) glDeleteTextures(1, &c->tex);
    caps[idx] = caps[--ncaps];
}

// Keep the pointer inside the region: a barrier on every region edge that is
// not also a screen edge (the screen edge already stops it).
static void put_up_barriers()
{
    int sw = DisplayWidth(dpy, DefaultScreen(dpy)), sh = DisplayHeight(dpy, DefaultScreen(dpy));
    int x0 = rx, y0 = ry, x1 = rx + rw, y1 = ry + rh;
    int edges[4][4] = { {x1, y0, x1, y1}, {x0, y0, x0, y1}, {x0, y1, x1, y1}, {x0, y0, x1, y0} };
    bool inside[4]  = { x1 < sw, x0 > 0, y1 < sh, y0 > 0 };
    for (int i = 0; i < 4; i++)
        if (inside[i])
            barriers[nbarriers++] = XFixesCreatePointerBarrier(
                dpy, root, edges[i][0], edges[i][1], edges[i][2], edges[i][3], 0, 0, nullptr);

    Window rr, cr; int px, py, wx, wy; unsigned int mask;
    if (XQueryPointer(dpy, root, &rr, &cr, &px, &py, &wx, &wy, &mask) &&
        (px < x0 || py < y0 || px >= x1 || py >= y1))
        XWarpPointer(dpy, None, root, 0, 0, 0, 0, x0 + rw / 2, y0 + rh / 2);
}

int cap_init(void *display, unsigned long self_xid, int x, int y, int w, int h)
{
    dpy = (Display *)display;
    if (!dpy) { fprintf(stderr, "[cap] null display\n"); return 0; }
    root = DefaultRootWindow(dpy);
    self_win = (Window)self_xid;
    rx = x; ry = y; rw = w; rh = h;

    int evb, erb, fmaj = 5, fmin = 0;
    if (!XCompositeQueryExtension(dpy, &evb, &erb)) { fprintf(stderr, "[cap] no Composite\n"); return 0; }
    if (!XFixesQueryExtension(dpy, &evb, &erb) || !XFixesQueryVersion(dpy, &fmaj, &fmin) || fmaj < 5) {
        fprintf(stderr, "[cap] need XFixes 5 (pointer barriers)\n"); return 0;
    }
    p_bind    = (PFNBindTexImageEXT)    glXGetProcAddress((const GLubyte *)"glXBindTexImageEXT");
    p_release = (PFNReleaseTexImageEXT) glXGetProcAddress((const GLubyte *)"glXReleaseTexImageEXT");
    if (!p_bind || !p_release) { fprintf(stderr, "[cap] no glXBindTexImageEXT\n"); return 0; }
    gather_candidates();
    if (ncand == 0) { fprintf(stderr, "[cap] no tfp FBConfigs\n"); return 0; }

    XSetErrorHandler(err_handler);

    // Our window goes inside the overlay window, above every other window, and
    // both let input fall through to the windows they cover. It moves before
    // the redirect, which would otherwise take it too as a child of the root.
    cow = XCompositeGetOverlayWindow(dpy, root);
    XserverRegion none = XFixesCreateRegion(dpy, nullptr, 0);
    XFixesSetWindowShapeRegion(dpy, cow, ShapeInput, 0, 0, none);
    XFixesSetWindowShapeRegion(dpy, self_win, ShapeInput, 0, 0, none);
    XFixesDestroyRegion(dpy, none);
    XReparentWindow(dpy, self_win, cow, 0, 0);
    XMapWindow(dpy, self_win);

    // Manual: X stops painting the windows at all; only we put pixels on screen.
    g_err = 0;
    XCompositeRedirectSubwindows(dpy, root, CompositeRedirectManual);
    XSync(dpy, False);
    if (g_err) {
        fprintf(stderr, "[cap] RedirectSubwindows failed (%d): another compositor running?\n", g_err);
        XCompositeReleaseOverlayWindow(dpy, root);
        cow = 0;
        return 0;
    }
    redirected = true;
    put_up_barriers();
    XSync(dpy, False);

    live = true;
    fprintf(stderr, "[cap] active; region %dx%d+%d+%d\n", rw, rh, rx, ry);
    return 1;
}

void cap_update()
{
    if (!live) return;
    Window r2, parent, *children = nullptr; unsigned int nch = 0;
    if (!XQueryTree(dpy, root, &r2, &parent, &children, &nch)) return;

    for (int i = 0; i < ncaps; i++) caps[i].seen = false;
    npub = 0;

    for (unsigned int i = 0; i < nch && npub < MAXWINS; i++) {
        Window w = children[i];
        if (w == self_win || w == cow) continue;
        XWindowAttributes wa;
        g_err = 0;
        if (!XGetWindowAttributes(dpy, w, &wa) || g_err) continue;
        if (wa.c_class != InputOutput || wa.map_state != IsViewable) continue;
        if (wa.x + wa.width <= rx || wa.y + wa.height <= ry || wa.x >= rx + rw || wa.y >= ry + rh) continue;

        Cap *c = find_cap(w);
        if (!c) { if (ncaps >= MAXWINS) continue; c = &caps[ncaps++]; *c = {}; c->xid = w; c->depth = wa.depth; }
        bool need = (!c->glx) || (wa.width != c->w) || (wa.height != c->h) || (wa.depth != c->depth);
        c->x = wa.x; c->y = wa.y; c->w = wa.width; c->h = wa.height; c->depth = wa.depth;
        if (need && !acquire(c)) { c->seen = true; continue; }
        c->seen = true;

        glBindTexture(GL_TEXTURE_2D, c->tex);
        if (c->bound) p_release(dpy, c->glx, GLX_FRONT_EXT);
        p_bind(dpy, c->glx, GLX_FRONT_EXT, NULL);
        c->bound = true;

        pub[npub++] = CapWin{ (unsigned long)w, c->tex, c->x, c->y, c->w, c->h };
    }
    glBindTexture(GL_TEXTURE_2D, 0);
    if (children) XFree(children);

    for (int i = ncaps - 1; i >= 0; i--) if (!caps[i].seen) drop_cap(i);
}

int cap_count()              { return npub; }
const CapWin *cap_windows()  { return pub; }
int cap_region_x()           { return rx; }
int cap_region_y()           { return ry; }
int cap_region_w()           { return rw; }
int cap_region_h()           { return rh; }

void cap_pointer(float *u, float *v, int *inside)
{
    if (inside) *inside = 0;
    if (!dpy) return;
    Window rr, cr; int px, py, wx, wy; unsigned int mask;
    if (!XQueryPointer(dpy, root, &rr, &cr, &px, &py, &wx, &wy, &mask)) return;
    if (px < rx || py < ry || px >= rx + rw || py >= ry + rh) return;
    if (u) *u = (px - rx) / (float)rw;
    if (v) *v = (py - ry) / (float)rh;
    if (inside) *inside = 1;
}

void cap_shutdown()
{
    if (!dpy) return;
    for (int i = ncaps - 1; i >= 0; i--) drop_cap(i);
    for (int i = 0; i < nbarriers; i++) XFixesDestroyPointerBarrier(dpy, barriers[i]);
    nbarriers = 0;
    if (redirected) XCompositeUnredirectSubwindows(dpy, root, CompositeRedirectManual);
    redirected = false;
    if (cow) {
        XUnmapWindow(dpy, self_win);
        XReparentWindow(dpy, self_win, root, 0, 0);
        XCompositeReleaseOverlayWindow(dpy, root);
        cow = 0;
    }
    XSync(dpy, False);
    live = false;
    dpy = nullptr;
}
