// window_capture.cpp — lightweight GPU compositor over X11.
// Pure X11/GLX (no raylib.h here). See window_capture.h.
//
// Per frame: enumerate top-level windows (XQueryTree, bottom-to-top), keep a GL
// texture per window via GLX_EXT_texture_from_pixmap (zero copy), and expose each
// with its screen rectangle. The renderer composites them into one panel.

#include "window_capture.h"

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/extensions/Xcomposite.h>
#include <X11/extensions/Xrandr.h>
#include <X11/extensions/Xfixes.h>
#include <X11/extensions/shape.h>   // ShapeInput
#include <GL/gl.h>
#include <GL/glx.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" Display *glfwGetX11Display(void);
extern "C" Window   glfwGetX11Window(void *glfw_window);

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
    bool      bound;   // tex image currently bound (release only if so)
};

static Display *dpy        = nullptr;
static Window   root       = 0;
static Window   self_win   = 0;     // our own raylib window (excluded)
static bool     live       = false;
static bool     redirected = false;
static bool     cursor_hidden = false;

static int rx = 0, ry = 0, rw = 0, rh = 0;  // capture region (root coords)

static Cap   caps[MAXWINS]; static int ncaps = 0;
static WCWin pub[MAXWINS];  static int npub  = 0;

// candidate tfp FBConfigs + per-depth winning choice
static GLXFBConfig cands[256]; static int ncand = 0;
struct DCache { int depth; GLXFBConfig fbc; int fmt; bool set; };
static DCache dcache[8]; static int ndc = 0;

static unsigned char g_err = 0;
static int err_handler(Display *, XErrorEvent *e) { g_err = e->error_code; return 0; }

static void gather_candidates(void)
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
    const int attrs[] = {
        GLX_TEXTURE_TARGET_EXT, GLX_TEXTURE_2D_EXT,
        GLX_TEXTURE_FORMAT_EXT, fmt,
        None
    };
    g_err = 0;
    GLXPixmap gp = glXCreatePixmap(dpy, fbc, pix, attrs);
    XSync(dpy, False);
    if (g_err) return 0;
    return gp;
}

// (Re)bind a window's offscreen pixmap to a GL texture.
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
    if (dc && dc->set) {
        c->glx = try_create(c->pix, dc->fbc, dc->fmt);
    }
    if (!c->glx) {  // discover a working FBConfig for this depth
        int fmts[2] = { GLX_TEXTURE_FORMAT_RGB_EXT, GLX_TEXTURE_FORMAT_RGBA_EXT };
        for (int ci = 0; ci < ncand && !c->glx; ci++)
            for (int fi = 0; fi < 2 && !c->glx; fi++) {
                GLXPixmap gp = try_create(c->pix, cands[ci], fmts[fi]);
                if (gp) { c->glx = gp; if (dc) { dc->fbc = cands[ci]; dc->fmt = fmts[fi]; dc->set = true; }
                          fprintf(stderr, "[wc] depth %d: FBConfig #%d format %s\n",
                                  c->depth, ci, fmts[fi]==GLX_TEXTURE_FORMAT_RGBA_EXT?"RGBA":"RGB"); }
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

static Cap *find_cap(Window w)
{
    for (int i = 0; i < ncaps; i++) if (caps[i].xid == w) return &caps[i];
    return nullptr;
}

static void drop_cap(int idx)
{
    Cap *c = &caps[idx];
    if (c->glx) glXDestroyPixmap(dpy, c->glx);
    if (c->pix) XFreePixmap(dpy, c->pix);
    if (c->tex) glDeleteTextures(1, &c->tex);
    caps[idx] = caps[--ncaps];
}

// Capture region = the whole root by default (so a scaled/enlarged framebuffer,
// e.g. a 4K workspace via `xrandr --scale`, is captured in full). Override with
// SWM_REGION="WxH+X+Y" to restrict to one screen in a multi-monitor setup.
static void compute_region(void)
{
    rx = 0; ry = 0;
    rw = DisplayWidth(dpy, DefaultScreen(dpy));
    rh = DisplayHeight(dpy, DefaultScreen(dpy));

    const char *env = getenv("SWM_REGION");
    if (env) {
        int w, h, x, y;
        if (sscanf(env, "%dx%d+%d+%d", &w, &h, &x, &y) == 4) { rw = w; rh = h; rx = x; ry = y; }
    }
}

bool wc_init(void *glfw_window_handle)
{
    dpy = glfwGetX11Display();
    if (!dpy) { fprintf(stderr, "[wc] no X display\n"); return false; }
    root = DefaultRootWindow(dpy);
    if (glfw_window_handle) self_win = glfwGetX11Window(glfw_window_handle);

    int evb, erb;
    if (!XCompositeQueryExtension(dpy, &evb, &erb)) { fprintf(stderr, "[wc] no Composite\n"); return false; }
    p_bind    = (PFNBindTexImageEXT)    glXGetProcAddress((const GLubyte *)"glXBindTexImageEXT");
    p_release = (PFNReleaseTexImageEXT) glXGetProcAddress((const GLubyte *)"glXReleaseTexImageEXT");
    if (!p_bind || !p_release) { fprintf(stderr, "[wc] no glXBindTexImageEXT\n"); return false; }

    // Install our non-fatal error handler for the whole app lifetime: a
    // compositor must survive windows being resized/destroyed under it, which
    // would otherwise abort the process via the default Xlib handler.
    XSetErrorHandler(err_handler);

    // Become the compositor: redirect all current+future top-levels (automatic
    // so the server still draws them to the laptop screen as well).
    g_err = 0;
    XCompositeRedirectSubwindows(dpy, root, CompositeRedirectAutomatic);
    XSync(dpy, False);
    if (g_err) {
        fprintf(stderr, "[wc] RedirectSubwindows failed (err %d) — another compositor (picom) running?\n", g_err);
        return false;
    }
    redirected = true;

    compute_region();
    gather_candidates();
    if (ncand == 0) { fprintf(stderr, "[wc] no tfp FBConfigs\n"); return false; }

    live = true;
    fprintf(stderr, "[wc] compositor active; region %dx%d+%d+%d, excluding self 0x%lx\n",
            rw, rh, rx, ry, (unsigned long)self_win);
    return true;
}

void wc_update(void)
{
    if (!live) return;

    Window r2, parent, *children = nullptr; unsigned int nch = 0;
    if (!XQueryTree(dpy, root, &r2, &parent, &children, &nch)) return;

    for (int i = 0; i < ncaps; i++) caps[i].seen = false;
    npub = 0;

    for (unsigned int i = 0; i < nch && npub < MAXWINS; i++) {  // bottom-to-top
        Window w = children[i];
        if (w == self_win) continue;

        XWindowAttributes wa;
        g_err = 0;
        if (!XGetWindowAttributes(dpy, w, &wa) || g_err) continue;
        if (wa.c_class != InputOutput) continue;
        if (wa.map_state != IsViewable) continue;
        // skip windows fully outside the capture region
        if (wa.x + wa.width <= rx || wa.y + wa.height <= ry ||
            wa.x >= rx + rw      || wa.y >= ry + rh) continue;

        Cap *c = find_cap(w);
        if (!c) {
            if (ncaps >= MAXWINS) continue;
            c = &caps[ncaps++];
            *c = {}; c->xid = w; c->depth = wa.depth;
        }
        bool need = (!c->glx) || (wa.width != c->w) || (wa.height != c->h) || (wa.depth != c->depth);
        c->x = wa.x; c->y = wa.y; c->w = wa.width; c->h = wa.height; c->depth = wa.depth;
        if (need && !acquire(c)) { c->seen = true; continue; }  // keep entry, skip this frame
        c->seen = true;

        // Refresh content (release only if previously bound). Errors here are
        // swallowed by the persistent handler; the window is re-acquired or
        // reaped on a later frame.
        glBindTexture(GL_TEXTURE_2D, c->tex);
        if (c->bound) p_release(dpy, c->glx, GLX_FRONT_EXT);
        p_bind(dpy, c->glx, GLX_FRONT_EXT, NULL);
        c->bound = true;

        pub[npub++] = WCWin{ (unsigned long)w, c->tex, c->x, c->y, c->w, c->h };
    }
    glBindTexture(GL_TEXTURE_2D, 0);

    if (children) XFree(children);

    // Reap windows that disappeared.
    for (int i = ncaps - 1; i >= 0; i--) if (!caps[i].seen) drop_cap(i);
}

void wc_setup_overlay(void *glfw_window_handle, int x, int y, int w, int h)
{
    if (!dpy || !glfw_window_handle) return;
    Window win = glfwGetX11Window(glfw_window_handle);
    self_win = win;

    // override-redirect must be set while unmapped to take effect.
    XUnmapWindow(dpy, win);
    XSync(dpy, False);
    XSetWindowAttributes a; a.override_redirect = True;
    XChangeWindowAttributes(dpy, win, CWOverrideRedirect, &a);
    XSetWindowBorderWidth(dpy, win, 0);   // drop any border the WM gave it
    XMoveResizeWindow(dpy, win, x, y, w, h);
    XMapRaised(dpy, win);
    XSync(dpy, False);

    // Empty input shape => pointer/keyboard pass through to windows behind.
    XserverRegion region = XFixesCreateRegion(dpy, NULL, 0);
    XFixesSetWindowShapeRegion(dpy, win, ShapeInput, 0, 0, region);
    XFixesDestroyRegion(dpy, region);

    // Hide the hardware cursor over our overlay (we draw our own on the plane).
    XFixesHideCursor(dpy, win);
    cursor_hidden = true;
    XSync(dpy, False);
}

bool wc_key_down(unsigned long keysym)
{
    if (!dpy) return false;
    KeyCode kc = XKeysymToKeycode(dpy, (KeySym)keysym);
    if (!kc) return false;
    char keys[32];
    XQueryKeymap(dpy, keys);
    return (keys[kc / 8] >> (kc % 8)) & 1;
}

bool wc_pointer(float *u, float *v)
{
    if (!dpy) return false;
    Window rr, cr; int px, py, wx, wy; unsigned int mask;
    if (!XQueryPointer(dpy, root, &rr, &cr, &px, &py, &wx, &wy, &mask)) return false;
    if (px < rx || py < ry || px >= rx + rw || py >= ry + rh) return false;
    *u = (px - rx) / (float)rw;
    *v = (py - ry) / (float)rh;
    return true;
}

void wc_raise_overlay(void)
{
    if (dpy && self_win) XRaiseWindow(dpy, self_win);
}

void wc_root_size(int *w, int *h)
{
    Display *d = dpy ? dpy : glfwGetX11Display();
    if (!d) { if (w) *w = 0; if (h) *h = 0; return; }
    int s = DefaultScreen(d);
    if (w) *w = DisplayWidth(d, s);
    if (h) *h = DisplayHeight(d, s);
}

int wc_screen_width(void)  { return rw; }
int wc_screen_height(void) { return rh; }
int wc_region_x(void)      { return rx; }
int wc_region_y(void)      { return ry; }
int wc_count(void)         { return npub; }
const WCWin *wc_windows(void) { return pub; }
bool wc_ok(void)           { return live && npub > 0; }

// ---- Mouse input (dedicated X connection + passive grabs) ------------------

static Display *in_dpy        = nullptr;
static Window   in_root       = 0;
static int      in_scroll     = 0;       // accumulated scroll ticks (+up/-down)
static bool     in_toggle     = false;   // toggle-FLAT/SBS button pressed
static bool     in_reset      = false;   // reset-zoom/pan button pressed
static int      in_drag_btn   = 0;       // button currently dragging (0 = none)
static int      in_last_x = 0, in_last_y = 0;
static float    in_pan_dx = 0, in_pan_dy = 0;  // middle-drag accumulator (pan)
static float    in_conv_dx = 0;                // left-drag x accumulator (convergence)
static int      in_reset_btn  = 9;       // side button: reset zoom+pan
static int      in_toggle_btn = 8;       // side button: toggle FLAT/SBS

bool wc_input_init(void)
{
    in_dpy = XOpenDisplay(nullptr);
    if (!in_dpy) { fprintf(stderr, "[wc] input: XOpenDisplay failed\n"); return false; }
    in_root = DefaultRootWindow(in_dpy);

    if (const char *e = getenv("SWM_BTN_RESET"))  { int b = atoi(e); if (b > 0) in_reset_btn  = b; }
    if (const char *e = getenv("SWM_BTN_TOGGLE")) { int b = atoi(e); if (b > 0) in_toggle_btn = b; }

    // Super+Alt as the chord; also grab with the lock modifiers so NumLock/CapsLock
    // don't defeat the match. Buttons: 1 = left-drag (convergence), 2 = middle-drag
    // (pan), 4/5 = scroll/zoom, plus the two side buttons (reset / toggle).
    unsigned int mod = Mod1Mask | Mod4Mask;             // Alt + Super
    unsigned int locks[] = { 0, LockMask, Mod2Mask, LockMask | Mod2Mask };
    int buttons[] = { 1, 2, 4, 5, in_reset_btn, in_toggle_btn };
    unsigned int mask = ButtonPressMask | ButtonReleaseMask | PointerMotionMask;
    for (int b : buttons)
        for (unsigned int l : locks)
            XGrabButton(in_dpy, b, mod | l, in_root, False, mask,
                        GrabModeAsync, GrabModeAsync, None, None);
    XSync(in_dpy, False);
    return true;
}

void wc_input_poll(void)
{
    if (!in_dpy) return;
    while (XPending(in_dpy)) {
        XEvent e;
        XNextEvent(in_dpy, &e);
        if (e.type == ButtonPress) {
            int b = e.xbutton.button;
            if      (b == 4) in_scroll += 1;            // scroll up
            else if (b == 5) in_scroll -= 1;            // scroll down
            else if (b == 1 || b == 2) {                // start left/middle drag
                in_drag_btn = b;
                in_last_x = e.xbutton.x_root; in_last_y = e.xbutton.y_root;
            }
            else if (b == in_reset_btn)  in_reset  = true;
            else if (b == in_toggle_btn) in_toggle = true;
        } else if (e.type == ButtonRelease) {
            if (e.xbutton.button == in_drag_btn) in_drag_btn = 0;
        } else if (e.type == MotionNotify) {
            if (in_drag_btn) {
                int dx = e.xmotion.x_root - in_last_x;
                int dy = e.xmotion.y_root - in_last_y;
                in_last_x = e.xmotion.x_root; in_last_y = e.xmotion.y_root;
                if      (in_drag_btn == 2) { in_pan_dx += dx; in_pan_dy += dy; }  // pan
                else if (in_drag_btn == 1) { in_conv_dx += dx; }                 // convergence
            }
        }
    }
}

int  wc_take_scroll(void) { int s = in_scroll; in_scroll = 0; return s; }
bool wc_take_toggle(void) { bool t = in_toggle; in_toggle = false; return t; }
bool wc_take_reset(void)  { bool r = in_reset;  in_reset  = false; return r; }
void wc_take_pan(float *dx, float *dy)
{
    if (dx) *dx = in_pan_dx;
    if (dy) *dy = in_pan_dy;
    in_pan_dx = in_pan_dy = 0.0f;
}
float wc_take_convx(void) { float v = in_conv_dx; in_conv_dx = 0.0f; return v; }

void wc_input_shutdown(void)
{
    if (!in_dpy) return;
    XUngrabButton(in_dpy, AnyButton, AnyModifier, in_root);
    XCloseDisplay(in_dpy);
    in_dpy = nullptr;
}

void wc_shutdown(void)
{
    if (!dpy) return;
    for (int i = ncaps - 1; i >= 0; i--) drop_cap(i);
    if (cursor_hidden && self_win) { XFixesShowCursor(dpy, self_win); cursor_hidden = false; }
    if (redirected) XCompositeUnredirectSubwindows(dpy, root, CompositeRedirectAutomatic);
    XSync(dpy, False);
    live = false;
}
