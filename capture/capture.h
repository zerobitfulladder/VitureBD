// capture.h — C-ABI desktop window capture into the CURRENT GL context.
//
// A small GPU compositor. It redirects the root's children (Composite, manual:
// X stops painting them) and exposes each top-level window as a GL texture via
// GLX_EXT_texture_from_pixmap, plus its screen rectangle. The caller's window
// moves inside the Composite overlay window, above everything, so it is the
// only thing on screen; both pass input through to the windows beneath, and
// pointer barriers keep the pointer inside the capture region. Because it reads
// each window's backing pixmap, never the screen, there is no feedback loop.
//
// The caller owns the X Display and the GL context: pass glfw's Display* and
// the window it presents in, and call cap_update() with that GL context
// current. The returned texture names are valid in the caller's context (wrap
// them with moderngl's external_texture). No other compositor may run (picom
// off). If the process dies, the X server drops the redirect and the overlay
// window itself, so the desktop comes back on its own.
#pragma once
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    unsigned long xid;   // X window id
    unsigned int  tex;   // GL texture name (valid in the caller's context)
    int x, y, w, h;      // geometry in root pixels
} CapWin;

// Become the compositor and start capturing. `dpy` is the caller's X Display*
// (e.g. glfw.get_x11_display()); `self_xid` is the caller's (unmapped) window,
// sized to the screen, which this moves into the overlay window and maps.
// region = the sub-rectangle of the root to capture (the desktop). Returns 1
// on success, 0 on failure (e.g. another compositor running).
int  cap_init(void *dpy, unsigned long self_xid, int rx, int ry, int rw, int rh);

// Refresh window list + textures. Call once per frame with the GL context current.
void cap_update(void);

int  cap_count(void);
const CapWin *cap_windows(void);   // array of cap_count(), bottom-to-top
int  cap_region_x(void);
int  cap_region_y(void);
int  cap_region_w(void);
int  cap_region_h(void);

// Real pointer position normalized in the region (u,v in 0..1); *inside=1 if
// the pointer is within the region.
void cap_pointer(float *u, float *v, int *inside);

void cap_shutdown(void);

#ifdef __cplusplus
}
#endif
