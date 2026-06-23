// capture_probe.c — de-risk: capture an X window's pixmap fully on the GPU via
// GLX_EXT_texture_from_pixmap and render it to a test window on :0.
//
// Purpose: prove the zero-copy capture primitive works on this NVIDIA + picom
// box before wiring it into the Viture 6DOF demo. No CPU readback anywhere.
//
// This version is EMPIRICAL about FBConfig selection: NVIDIA throws BadAlloc on
// glXCreatePixmap for configs it doesn't like, and the exact rule is fiddly, so
// we install a non-fatal X error handler and trial-create the GLXPixmap against
// every candidate (config x texture-format) until one succeeds.
//
// Usage:
//   ./capture_probe              # capture the currently active window
//   ./capture_probe 0x1c00004    # capture a specific window id (hex or dec)
//   ./capture_probe overlay      # capture the composite overlay window (whole desktop)
//
// Controls: ESC/q to quit. The captured window updates live.

#include <X11/Xlib.h>
#include <X11/Xatom.h>
#include <X11/keysym.h>
#include <X11/extensions/Xcomposite.h>
#include <GL/gl.h>
#include <GL/glx.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// --- GLX texture_from_pixmap entry points (resolved at runtime) --------------
typedef void (*PFNBindTexImageEXT)(Display *, GLXDrawable, int, const int *);
typedef void (*PFNReleaseTexImageEXT)(Display *, GLXDrawable, int);
static PFNBindTexImageEXT    p_glXBindTexImageEXT;
static PFNReleaseTexImageEXT p_glXReleaseTexImageEXT;

static Display *dpy;
static int      screen;
static Window   root;

// Non-fatal X error handler: record the last error instead of exiting, so we can
// probe glXCreatePixmap against multiple FBConfigs.
static unsigned char g_last_error = 0;
static int x_error_handler(Display *d, XErrorEvent *e) {
    (void)d; g_last_error = e->error_code; return 0;
}

static void die(const char *msg) { fprintf(stderr, "FATAL: %s\n", msg); exit(1); }

static Window active_window(void) {
    Atom prop = XInternAtom(dpy, "_NET_ACTIVE_WINDOW", True);
    if (prop == None) return None;
    Atom at; int af; unsigned long n, after; unsigned char *data = NULL;
    if (XGetWindowProperty(dpy, root, prop, 0, 1, False, AnyPropertyType,
                           &at, &af, &n, &after, &data) != Success) return None;
    Window w = None;
    if (data && n >= 1) w = *(Window *)data;
    if (data) XFree(data);
    return w;
}

// Double-buffered RGBA FBConfig for our own output window.
static GLXFBConfig choose_window_fbconfig(XVisualInfo **out_vi) {
    int attrs[] = {
        GLX_X_RENDERABLE, True,
        GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT,
        GLX_RENDER_TYPE,   GLX_RGBA_BIT,
        GLX_DOUBLEBUFFER,  True,
        GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8,
        None
    };
    int n = 0;
    GLXFBConfig *cfgs = glXChooseFBConfig(dpy, screen, attrs, &n);
    if (!cfgs || n == 0) die("no FBConfig for output window");
    GLXFBConfig fbc = cfgs[0];
    *out_vi = glXGetVisualFromFBConfig(dpy, fbc);
    XFree(cfgs);
    return fbc;
}

// --- candidate texture_from_pixmap FBConfigs ---------------------------------
static GLXFBConfig g_cands[256];
static int         g_ncand = 0;

// Winning combination, discovered by trial on first acquire.
static GLXFBConfig g_win_fbc = NULL;
static int         g_win_fmt = 0;     // GLX_TEXTURE_FORMAT_RGB/RGBA_EXT

// Gather every FBConfig that can be bound as a 2D texture from a pixmap.
static void gather_candidates(void) {
    int n = 0;
    GLXFBConfig *cfgs = glXGetFBConfigs(dpy, screen, &n);
    if (!cfgs || n == 0) die("glXGetFBConfigs returned none");
    for (int i = 0; i < n && g_ncand < 256; i++) {
        int draw = 0, targets = 0, rgb = 0, rgba = 0;
        glXGetFBConfigAttrib(dpy, cfgs[i], GLX_DRAWABLE_TYPE, &draw);
        if (!(draw & GLX_PIXMAP_BIT)) continue;
        glXGetFBConfigAttrib(dpy, cfgs[i], GLX_BIND_TO_TEXTURE_TARGETS_EXT, &targets);
        if (!(targets & GLX_TEXTURE_2D_BIT_EXT)) continue;
        glXGetFBConfigAttrib(dpy, cfgs[i], GLX_BIND_TO_TEXTURE_RGB_EXT,  &rgb);
        glXGetFBConfigAttrib(dpy, cfgs[i], GLX_BIND_TO_TEXTURE_RGBA_EXT, &rgba);
        if (!rgb && !rgba) continue;
        g_cands[g_ncand++] = cfgs[i];
    }
    XFree(cfgs);
    printf("gathered %d texture_from_pixmap-capable FBConfigs\n", g_ncand);
    if (g_ncand == 0) die("no tfp-capable FBConfigs at all");
}

// Try to create a GLXPixmap for `pix` with a given (fbc, fmt). Returns the
// GLXPixmap on success, or 0 (no X error left pending) on failure.
static GLXPixmap try_create(Pixmap pix, GLXFBConfig fbc, int fmt) {
    const int attrs[] = {
        GLX_TEXTURE_TARGET_EXT, GLX_TEXTURE_2D_EXT,
        GLX_TEXTURE_FORMAT_EXT, fmt,
        None
    };
    g_last_error = 0;
    GLXPixmap gp = glXCreatePixmap(dpy, fbc, pix, attrs);
    XSync(dpy, False);            // flush so any async error reaches our handler
    if (g_last_error) return 0;   // failed; leak the id (resource wasn't created)
    return gp;
}

int main(int argc, char **argv) {
    dpy = XOpenDisplay(NULL);
    if (!dpy) die("cannot open display :0");
    screen = DefaultScreen(dpy);
    root   = RootWindow(dpy, screen);
    XSetErrorHandler(x_error_handler);

    int evb, erb;
    if (!XCompositeQueryExtension(dpy, &evb, &erb)) die("Composite extension missing");
    int cmaj = 0, cmin = 0; XCompositeQueryVersion(dpy, &cmaj, &cmin);
    printf("Composite extension %d.%d\n", cmaj, cmin);

    // ---- capture target -----------------------------------------------------
    Window target = None; int capture_overlay = 0;
    if (argc >= 2 && strcmp(argv[1], "overlay") == 0) {
        target = XCompositeGetOverlayWindow(dpy, root); capture_overlay = 1;
        printf("Target: composite overlay 0x%lx (whole desktop)\n", target);
    } else if (argc >= 2) {
        target = (Window)strtoul(argv[1], NULL, 0);
        printf("Target: window 0x%lx (arg)\n", target);
    } else {
        target = active_window();
        printf("Target: active window 0x%lx\n", target);
    }
    if (target == None) die("no capture target");

    // Redirect the target ourselves (automatic mode = still drawn to screen).
    // Harmless extra reference if a compositor already redirects it; required if
    // no compositor is running.  Tolerate an error (e.g. already manually redirected).
    int we_redirected = 0;
    if (!capture_overlay) {
        g_last_error = 0;
        XCompositeRedirectWindow(dpy, target, CompositeRedirectAutomatic);
        XSync(dpy, False);
        if (g_last_error)
            printf("redirect: X error %d (relying on existing redirect)\n", g_last_error);
        else { we_redirected = 1; printf("redirect: target redirected (automatic)\n"); }
    }

    XWindowAttributes wa;
    if (!XGetWindowAttributes(dpy, target, &wa)) die("XGetWindowAttributes failed");
    int tw = wa.width, th = wa.height;
    printf("Target geometry: %dx%d depth=%d\n", tw, th, wa.depth);
    XSelectInput(dpy, target, StructureNotifyMask);

    // ---- output window + GL context ----------------------------------------
    XVisualInfo *vi = NULL;
    GLXFBConfig win_fbc = choose_window_fbconfig(&vi);
    if (!vi) die("no visual for output window");
    Colormap cmap = XCreateColormap(dpy, root, vi->visual, AllocNone);
    XSetWindowAttributes swa;
    swa.colormap = cmap; swa.background_pixel = 0; swa.border_pixel = 0;
    swa.event_mask = ExposureMask | KeyPressMask | StructureNotifyMask;
    int ow = tw > 1280 ? 1280 : tw, oh = th > 800 ? 800 : th;
    if (ow < 320) ow = 640;
    if (oh < 240) oh = 480;
    Window out = XCreateWindow(dpy, root, 0, 0, ow, oh, 0, vi->depth, InputOutput,
                               vi->visual, CWColormap|CWBackPixel|CWBorderPixel|CWEventMask, &swa);
    XStoreName(dpy, out, "capture_probe - texture_from_pixmap");
    XMapWindow(dpy, out);

    GLXContext ctx = glXCreateNewContext(dpy, win_fbc, GLX_RGBA_TYPE, NULL, True);
    if (!ctx) die("glXCreateNewContext failed");
    if (!glXMakeCurrent(dpy, out, ctx)) die("glXMakeCurrent failed");
    p_glXBindTexImageEXT    = (PFNBindTexImageEXT)    glXGetProcAddress((const GLubyte*)"glXBindTexImageEXT");
    p_glXReleaseTexImageEXT = (PFNReleaseTexImageEXT) glXGetProcAddress((const GLubyte*)"glXReleaseTexImageEXT");
    if (!p_glXBindTexImageEXT || !p_glXReleaseTexImageEXT) die("glXBindTexImageEXT missing");
    printf("GL_RENDERER: %s\n", (const char*)glGetString(GL_RENDERER));

    gather_candidates();

    GLuint tex = 0; glGenTextures(1, &tex);

    Pixmap    named_pix = 0;
    GLXPixmap glx_pix   = 0;
    int cur_w = 0, cur_h = 0;

    // Acquire (or re-acquire after resize) the named pixmap + GLXPixmap.
    // First call trials all candidates to find a working (fbconfig, format).
    #define ACQUIRE() do {                                                        \
        if (glx_pix)   { glXDestroyPixmap(dpy, glx_pix); glx_pix = 0; }            \
        if (named_pix) { XFreePixmap(dpy, named_pix);     named_pix = 0; }          \
        g_last_error = 0;                                                          \
        named_pix = XCompositeNameWindowPixmap(dpy, target);                       \
        XSync(dpy, False);                                                         \
        if (g_last_error) { fprintf(stderr, "NameWindowPixmap X error %d\n",       \
                                    g_last_error); die("cannot name pixmap"); }     \
        if (g_win_fbc) {                                                           \
            glx_pix = try_create(named_pix, g_win_fbc, g_win_fmt);                 \
        } else {                                                                   \
            int fmts[2] = { GLX_TEXTURE_FORMAT_RGB_EXT, GLX_TEXTURE_FORMAT_RGBA_EXT };\
            for (int ci = 0; ci < g_ncand && !glx_pix; ci++)                       \
              for (int fi = 0; fi < 2 && !glx_pix; fi++) {                         \
                GLXPixmap gp = try_create(named_pix, g_cands[ci], fmts[fi]);       \
                if (gp) { glx_pix = gp; g_win_fbc = g_cands[ci]; g_win_fmt = fmts[fi];\
                          printf("WORKS: candidate #%d, format %s\n", ci,          \
                                 fmts[fi]==GLX_TEXTURE_FORMAT_RGBA_EXT?"RGBA":"RGB");}\
              }                                                                    \
        }                                                                         \
        XGetWindowAttributes(dpy, target, &wa); cur_w = wa.width; cur_h = wa.height;\
    } while (0)

    XSync(dpy, False);
    ACQUIRE();
    if (!glx_pix) die("no FBConfig accepted glXCreatePixmap (capture not possible)");
    printf("Acquired: pixmap 0x%lx -> GLXPixmap 0x%lx\n", named_pix, glx_pix);

    glEnable(GL_TEXTURE_2D);
    glClearColor(0.05f, 0.05f, 0.08f, 1.0f);

    int running = 1, win_w = ow, win_h = oh;
    unsigned long frames = 0;
    while (running) {
        while (XPending(dpy)) {
            XEvent ev; XNextEvent(dpy, &ev);
            if (ev.type == KeyPress) {
                KeySym ks = XLookupKeysym(&ev.xkey, 0);
                if (ks == XK_Escape || ks == 'q') running = 0;
            } else if (ev.type == ConfigureNotify) {
                if (ev.xconfigure.window == out) {
                    win_w = ev.xconfigure.width; win_h = ev.xconfigure.height;
                } else if (ev.xconfigure.window == target &&
                           (ev.xconfigure.width != cur_w || ev.xconfigure.height != cur_h)) {
                    ACQUIRE();
                }
            } else if (ev.type == DestroyNotify && ev.xdestroywindow.window == target) {
                printf("target destroyed\n"); running = 0;
            }
        }

        glBindTexture(GL_TEXTURE_2D, tex);
        p_glXReleaseTexImageEXT(dpy, glx_pix, GLX_FRONT_EXT);
        p_glXBindTexImageEXT(dpy, glx_pix, GLX_FRONT_EXT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glViewport(0, 0, win_w, win_h);
        glClear(GL_COLOR_BUFFER_BIT);
        glBegin(GL_QUADS);            // flip V: X pixmaps are top-down
            glTexCoord2f(0, 1); glVertex2f(-1, -1);
            glTexCoord2f(1, 1); glVertex2f( 1, -1);
            glTexCoord2f(1, 0); glVertex2f( 1,  1);
            glTexCoord2f(0, 0); glVertex2f(-1,  1);
        glEnd();
        glXSwapBuffers(dpy, out);
        frames++;
    }

    printf("rendered %lu frames\n", frames);
    if (glx_pix)   glXDestroyPixmap(dpy, glx_pix);
    if (named_pix) XFreePixmap(dpy, named_pix);
    glXMakeCurrent(dpy, 0, 0);
    glXDestroyContext(dpy, ctx);
    XDestroyWindow(dpy, out);
    if (we_redirected) XCompositeUnredirectWindow(dpy, target, CompositeRedirectAutomatic);
    if (capture_overlay) XCompositeReleaseOverlayWindow(dpy, target);
    XCloseDisplay(dpy);
    return 0;
}
