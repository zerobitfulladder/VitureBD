// window_capture.h — lightweight GPU compositor: captures every top-level X11
// window via GLX_EXT_texture_from_pixmap and exposes each as a GL texture plus
// its on-screen rectangle, so the renderer can lay them out as one "desktop".
//
// Free of X11/raylib types so it can sit next to raylib.h (which collides with
// X11's `Font`). The .cpp owns all the X11.
//
// Requirements: runs on raylib's X display (:0) with NO other compositor (picom)
// redirecting the same windows — NVIDIA refuses a second GLXPixmap otherwise.
#pragma once

// One captured top-level window. `tex` is a GL texture name; x/y/w/h are the
// window's geometry in root (screen) pixels.
struct WCWin {
    unsigned long xid;
    unsigned int  tex;
    int x, y, w, h;
};

// Become a lightweight compositor and start capturing top-level windows.
// `glfw_window_handle` is raylib's GetWindowHandle() (a GLFWwindow*) so we can
// exclude our own window. Call AFTER InitWindow(). Returns false on failure.
bool wc_init(void *glfw_window_handle);

// Refresh the window list + textures. Call once per frame (GL context current).
void wc_update(void);

int  wc_screen_width(void);    // capture region width  (px)
int  wc_screen_height(void);   // capture region height (px)
int  wc_region_x(void);        // capture region origin in root coords
int  wc_region_y(void);

// Full X root (screen) size in pixels — the whole panel, independent of the
// capture region. Valid after InitWindow (uses the GLFW X display); does not
// require wc_init. Use this, not the RandR monitor size, to cover the panel:
// the CRTC mode can disagree with the framebuffer/screen size.
void wc_root_size(int *w, int *h);

int  wc_count(void);                 // number of captured windows
const struct WCWin *wc_windows(void); // array of wc_count(), bottom-to-top order
bool wc_ok(void);

// Real X pointer position normalized within the capture region (u,v in 0..1).
// Returns false if the pointer is outside the region. Used to draw the cursor
// on the panel so the real mouse/keyboard remain usable through the glasses.
bool wc_pointer(float *u, float *v);

// X KeySym values, exposed so callers needn't include X11 headers.
#define WC_KEY_F1  0xffbeUL
#define WC_KEY_F2  0xffbfUL
#define WC_KEY_F3  0xffc0UL
#define WC_KEY_F4  0xffc1UL
#define WC_KEY_F5  0xffc2UL
#define WC_KEY_F6  0xffc3UL
#define WC_KEY_ESC 0xff1bUL

// Turn our window into an override-redirect, input-transparent overlay at the
// given screen rectangle: the WM ignores it (no tiling fight) and mouse/keyboard
// pass through to the windows behind. Call after wc_init().
void wc_setup_overlay(void *glfw_window_handle, int x, int y, int w, int h);

// Raise the overlay to the top of the stacking order. Other windows (e.g. a
// fullscreen video) can raise themselves above us; call this each frame so our
// composited view stays visible on screen. No-op if no overlay window.
void wc_raise_overlay(void);

// ---- Mouse input -----------------------------------------------------------
// A dedicated X connection passively grabs Super+Alt + {drag, scroll, side
// button} so those combos are consumed (apps never see them) while all bare
// mouse input passes through untouched. Call wc_input_init() after wc_init();
// call wc_input_poll() once per frame, then drain with the take_* helpers.
bool wc_input_init(void);
void wc_input_poll(void);
int   wc_take_scroll(void);               // net scroll ticks since last call (+up/-down)
bool  wc_take_toggle(void);               // true once per Super+Alt toggle side-button press
bool  wc_take_reset(void);                // true once per Super+Alt reset side-button press
void  wc_take_pan(float *dx, float *dy);  // delta during Super+Alt+middle-drag (pan)
float wc_take_convx(void);                // x delta during Super+Alt+left-drag (convergence)
void  wc_input_shutdown(void);

// Focus-independent physical key state via XQueryKeymap (works even though the
// overlay receives no key events). True while the given key is held.
bool wc_key_down(unsigned long keysym);

void wc_shutdown(void);
