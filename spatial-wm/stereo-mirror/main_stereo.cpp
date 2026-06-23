// stereo-mirror — duplicate a 1920x1080 desktop into both eyes of the glasses
// while they are in 3D side-by-side mode (3840x1080), and split stereo SBS media
// so each eye sees its own half un-squished.
//
// NO device/IMU/pose/SDK code. The glasses must already be in 3D mode (set it
// with the hardware button / official app). This program only:
//   1. captures the desktop (left half of the 3840-wide root) via the shared
//      GPU compositor in ../glasses-demo/window_capture.cpp,
//   2. draws it twice — once into each 1920-wide eye half of the panel,
//   3. on the SBS toggle, splits the captured frame: left half -> left eye, right
//      half -> right eye, at native size, with zoom / pan / convergence tuning.
//
// The window is an override-redirect, input-transparent overlay: qtile leaves it
// alone and the real mouse/keyboard pass through to the desktop behind. Controls
// are read via XQueryKeymap and ALL require a Super+Alt chord, so bare keys never
// trigger us and the combos don't disturb qtile or focused apps. See README.
//
// Layout the desktop in a 1920x1080 box: set qtile's fake_screens to
// Screen(x=0, y=0, width=1920, height=1080). The right half of the 3840 root is
// dead space we never show directly (this overlay paints over the whole panel).

#include <atomic>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "raylib.h"
#include "rlgl.h"
#include "window_capture.h"

// X keysyms. All controls require the Super+Alt chord held (checked via
// XQueryKeymap), so bare keys never trigger us, and our combos are ignored by
// qtile (which never binds Super+Alt) and by apps (which don't use Super).
#define K_SUPER_L 0xffebUL
#define K_SUPER_R 0xffecUL
#define K_ALT_L   0xffe9UL
#define K_ALT_R   0xffeaUL
#define K_Q       0x0071UL   // quit
#define K_S       0x0073UL   // toggle SBS
#define K_W       0x0077UL   // swap eyes
#define K_0       0x0030UL   // reset view
#define K_EQUAL   0x003dUL   // zoom in
#define K_MINUS   0x002dUL   // zoom out
#define K_LEFT    0xff51UL   // pan left
#define K_UP      0xff52UL   // pan up
#define K_RIGHT   0xff53UL   // pan right
#define K_DOWN    0xff54UL   // pan down
#define K_LBRACK  0x005bUL   // convergence - (SBS)
#define K_RBRACK  0x005dUL   // convergence + (SBS)
#define K_SEMI    0x003bUL   // vertical align - (SBS)
#define K_APOS    0x0027UL   // vertical align + (SBS)

static std::atomic<bool> g_running{true};
static void on_signal(int) { g_running = false; }

// Build a raylib Texture2D view over a raw GL texture name owned by the
// compositor, so we can draw it with DrawTexturePro. Only id/width/height matter
// for drawing; format/mipmaps are placeholders.
static Texture2D wrap_tex(unsigned int id, int w, int h)
{
    Texture2D t;
    t.id      = id;
    t.width   = w;
    t.height  = h;
    t.mipmaps = 1;
    t.format  = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8;
    return t;
}

int main()
{
    std::signal(SIGINT, on_signal);

    SetTraceLogLevel(LOG_WARNING);

    // Panel size: the glasses in 3D SBS mode are a 3840-wide output, 1080 tall
    // (sometimes 1200). Hard-coded; override with SWM_PANEL="WxH" (e.g. 3840x1200)
    // when you switch the vertical mode.
    int gw = 3840, gh = 1080;
    if (const char *e = getenv("SWM_PANEL")) {
        int w, h; if (sscanf(e, "%dx%d", &w, &h) == 2) { gw = w; gh = h; }
    }

    // RESIZABLE is essential: a fixed-size GLFW window pins WM size hints and
    // snaps the framebuffer back to the initial size after our override-redirect
    // resize — that's what left the right eye unpainted, intermittently. With it,
    // GLFW tracks the resize to the full panel. UNDECORATED drops the WM frame in
    // the brief moment before we go override-redirect. Start at the panel size.
    SetConfigFlags(FLAG_VSYNC_HINT | FLAG_WINDOW_RESIZABLE | FLAG_WINDOW_UNDECORATED);
    InitWindow(gw, gh, "stereo-mirror");
    fprintf(stderr, "[stereo] panel %dx%d (SWM_PANEL to override)\n", gw, gh);

    const int eyeW = gw / 2;     // each eye = half the SBS panel
    const int eyeH = gh;

    // Capture the left eye box (the usable 1920xH desktop, where windows live —
    // including a "fullscreen" browser video, which stays 1920 wide). FLAT mirrors
    // this whole box to both eyes; SBS splits it into two 960 halves (half-SBS) and
    // un-squishes each to fill an eye. window_capture reads SWM_REGION.
    if (!getenv("SWM_REGION")) {
        char buf[64];
        snprintf(buf, sizeof(buf), "%dx%d+0+0", eyeW, eyeH);
        setenv("SWM_REGION", buf, 0);
    }

    bool overlay = !getenv("SWM_NO_OVERLAY");

    // Become the compositor and start capturing (needs picom off; run.sh handles).
    bool capture_ok = wc_init(GetWindowHandle());

    if (overlay) {
        // Override-redirect overlay covering the WHOLE panel; input-transparent so
        // the desktop stays usable through it.
        wc_setup_overlay(GetWindowHandle(), 0, 0, gw, gh);
        // GLFW does NOT track the raw-Xlib resize done inside wc_setup_overlay, so
        // its cached size is still the InitWindow size and raylib renders into that
        // tiny viewport (everything else shows the bare root). SetWindowSize now
        // corrects GLFW's cache and raylib's framebuffer to the full panel; the
        // window is already override-redirect, so the WM won't fight it.
        SetWindowSize(gw, gh);
        fprintf(stderr, "[stereo] overlay %dx%d, eyes %dx%d, region %s\n",
                gw, gh, eyeW, eyeH, getenv("SWM_REGION"));
    } else {
        SetWindowSize(gw, gh);                       // dev: plain window
        fprintf(stderr, "[stereo] dev windowed %dx%d\n", gw, gh);
    }

    // Mouse controls via passive Super+Alt grabs (drag = pan, scroll = zoom,
    // side button = toggle SBS). Consumed so apps don't see them; bare clicks pass.
    if (overlay) wc_input_init();

    const int rw = wc_screen_width()  > 0 ? wc_screen_width()  : eyeW;
    const int rh = wc_screen_height() > 0 ? wc_screen_height() : eyeH;
    const int rx = wc_region_x();
    const int ry = wc_region_y();

    // Offscreen desktop: composite the captured region here once per frame, then
    // blit it into the two eye halves.
    RenderTexture2D rt = LoadRenderTexture(rw, rh);
    SetTextureFilter(rt.texture, TEXTURE_FILTER_BILINEAR);  // smooth the 2x stretch

    // Stereo state.
    bool  sbs  = false;     // split captured frame per eye (SBS)
    bool  swap = false;     // swap L/R halves (cross-eyed content)
    float conv = 0.0f;      // horizontal convergence shift (px), SBS only
    float voff = 0.0f;      // vertical alignment offset (px), SBS only
    float zoom = 1.0f;      // magnification (both modes)
    float panx = 0.0f;      // pan offset x (px, both modes)
    float pany = 0.0f;      // pan offset y (px, both modes)
    if (const char *e = getenv("SWM_CONV")) conv = (float)atof(e);

    bool pQ = false, pS = false, pW = false, p0 = false;  // chord edge state

    int diag = 0;
    while (!WindowShouldClose() && g_running) {
        float dt = GetFrameTime();

        // One-time confirmation that the render target spans the whole panel.
        if (diag < 5) {
            int rW = 0, rH = 0; wc_root_size(&rW, &rH);
            fprintf(stderr, "[stereo] frame %d: render %dx%d root %dx%d (want %dx%d)\n",
                    diag, GetRenderWidth(), GetRenderHeight(), rW, rH, gw, gh);
        }
        diag++;

        // --- input: every control requires the Super+Alt chord ---------------
        bool mod = (wc_key_down(K_SUPER_L) || wc_key_down(K_SUPER_R)) &&
                   (wc_key_down(K_ALT_L)   || wc_key_down(K_ALT_R));

        // Edge-triggered toggles.
        bool cQ = mod && wc_key_down(K_Q);
        bool cS = mod && wc_key_down(K_S);
        bool cW = mod && wc_key_down(K_W);
        bool c0 = mod && wc_key_down(K_0);
        if (cQ && !pQ) break;                                   // quit
        if (cS && !pS) sbs  = !sbs;                             // toggle SBS
        if (cW && !pW) swap = !swap;                            // swap eyes
        if (c0 && !p0) { zoom = 1.0f; panx = pany = 0.0f; conv = voff = 0.0f; }
        pQ = cQ; pS = cS; pW = cW; p0 = c0;

        // Held (continuous) controls.
        if (mod) {
            float step = 220.0f * dt, zstep = 0.9f * dt;
            if (wc_key_down(K_EQUAL)) zoom += zstep * zoom;     // zoom in/out
            if (wc_key_down(K_MINUS)) zoom -= zstep * zoom;
            if (zoom < 0.2f) zoom = 0.2f;
            if (zoom > 8.0f) zoom = 8.0f;
            if (wc_key_down(K_LEFT))  panx -= step;             // pan
            if (wc_key_down(K_RIGHT)) panx += step;
            if (wc_key_down(K_UP))    pany -= step;
            if (wc_key_down(K_DOWN))  pany += step;
            if (wc_key_down(K_LBRACK)) conv -= step;            // convergence (SBS)
            if (wc_key_down(K_RBRACK)) conv += step;
            if (wc_key_down(K_SEMI))  voff -= step;             // vertical align (SBS)
            if (wc_key_down(K_APOS))  voff += step;
        }

        // Mouse (Super+Alt): scroll = zoom, middle-drag = pan, left-drag x =
        // convergence, side button = toggle SBS.
        wc_input_poll();
        if (wc_take_toggle()) sbs = !sbs;
        if (wc_take_reset())  { zoom = 1.0f; panx = pany = 0.0f; }   // side button: reset zoom+pan
        if (int sc = wc_take_scroll()) {
            zoom *= powf(1.1f, (float)sc);
            if (zoom < 0.2f) zoom = 0.2f;
            if (zoom > 8.0f) zoom = 8.0f;
        }
        float pdx = 0.0f, pdy = 0.0f;
        wc_take_pan(&pdx, &pdy);
        panx += pdx; pany += pdy;
        conv += wc_take_convx();

        // --- refresh capture -------------------------------------------------
        if (capture_ok) wc_update();

        // Keep our overlay on top: a fullscreen video raises its window above us
        // and would otherwise cover one eye on screen (our framebuffer is fine, but
        // it's hidden underneath). Re-raise every frame so our view stays visible.
        if (overlay) wc_raise_overlay();

        // --- composite desktop into the offscreen target ---------------------
        // Windows are drawn upright (tfp pixmaps are top-down, which matches
        // raylib's top-left texture origin). The RT itself is bottom-up, so the
        // final blit below flips V with a negative source height.
        BeginTextureMode(rt);
            ClearBackground(BLACK);
            if (capture_ok) {
                const WCWin *wins = wc_windows();
                for (int i = 0, n = wc_count(); i < n; i++) {
                    const WCWin &w = wins[i];
                    if (w.tex == 0 || w.w <= 0 || w.h <= 0) continue;
                    Texture2D t = wrap_tex(w.tex, w.w, w.h);
                    Rectangle src = { 0, 0, (float)w.w, (float)w.h };
                    Rectangle dst = { (float)(w.x - rx), (float)(w.y - ry),
                                      (float)w.w, (float)w.h };
                    DrawTexturePro(t, src, dst, {0, 0}, 0.0f, WHITE);
                }
                // Cursor: draw the real pointer (hardware cursor is hidden) so the
                // mouse stays usable through the glasses.
                float cu, cv;
                if (wc_pointer(&cu, &cv)) {
                    // Scale the crosshair by 1/zoom so it stays a constant size on
                    // screen after the zoomed blit.
                    float cx = cu * rw, cy = cv * rh;
                    float a = 14.0f / zoom, th = 2.0f / zoom, o = 1.0f / zoom;
                    DrawRectangleRec({ cx - a - o, cy - th - o, 2*a + 2*o, 2*th + 2*o }, BLACK);
                    DrawRectangleRec({ cx - th - o, cy - a - o, 2*th + 2*o, 2*a + 2*o }, BLACK);
                    DrawRectangleRec({ cx - a, cy - th, 2*a, 2*th }, GREEN);
                    DrawRectangleRec({ cx - th, cy - a, 2*th, 2*a }, GREEN);
                }
            }
        EndTextureMode();

        // --- present to both eyes -------------------------------------------
        // GLFW's cached window size is unreliable after the raw-Xlib overlay
        // resize (it snaps back to the InitWindow size), so don't trust raylib's
        // framebuffer size. The GLX drawable IS the real window (gw x gh), so we
        // drive the viewport + 2D projection to the panel ourselves every frame.
        BeginDrawing();
            rlDrawRenderBatchActive();               // flush raylib's setup batch
            rlViewport(0, 0, gw, gh);
            rlMatrixMode(RL_PROJECTION);
            rlPushMatrix();
            rlLoadIdentity();
            rlOrtho(0, gw, gh, 0, 0.0, 1.0);         // top-left origin (raylib 2D)
            rlMatrixMode(RL_MODELVIEW);
            rlLoadIdentity();

            ClearBackground(BLACK);

            // Zoom and pan apply to both modes. Each eye is scissored to its half
            // (raw GL coords, y from bottom) so magnified/panned content can't bleed
            // across the seam.
            rlEnableScissorTest();
            if (!sbs) {
                // FLAT: mirror the whole captured 1920 box into both eyes.
                Rectangle src = { 0, 0, (float)rw, -(float)rh };       // V-flip the RT
                float w = eyeW * zoom, h = eyeH * zoom;
                float x0 = ((float)eyeW - w) * 0.5f + panx;
                float y0 = ((float)eyeH - h) * 0.5f + pany;
                rlScissor(0, 0, eyeW, gh);
                DrawTexturePro(rt.texture, src, { x0, y0, w, h }, {0,0}, 0, WHITE);
                rlDrawRenderBatchActive();
                rlScissor(eyeW, 0, eyeW, gh);
                DrawTexturePro(rt.texture, src, { (float)eyeW + x0, y0, w, h }, {0,0}, 0, WHITE);
                rlDrawRenderBatchActive();
            } else {
                // SBS: two 960-wide views (native aspect, NOT anamorphic) — one per
                // eye, no stretch. zoom scales, pan offsets both eyes together, conv
                // is horizontal disparity, voff is vertical alignment.
                float half = rw * 0.5f;                                // 960
                float w = half * zoom, h = eyeH * zoom;
                float x0 = ((float)eyeW - w) * 0.5f + panx;
                float y0 = ((float)eyeH - h) * 0.5f + pany;
                Rectangle srcL = { swap ? half : 0.0f, 0, half, -(float)rh };
                Rectangle srcR = { swap ? 0.0f : half, 0, half, -(float)rh };
                rlScissor(0, 0, eyeW, gh);
                DrawTexturePro(rt.texture, srcL,
                               { x0 + conv*0.5f, y0 + voff*0.5f, w, h }, {0,0}, 0, WHITE);
                rlDrawRenderBatchActive();
                rlScissor(eyeW, 0, eyeW, gh);
                DrawTexturePro(rt.texture, srcR,
                               { (float)eyeW + x0 - conv*0.5f, y0 - voff*0.5f, w, h }, {0,0}, 0, WHITE);
                rlDrawRenderBatchActive();
            }
            rlDisableScissorTest();

            // Small status readout, drawn in BOTH eyes so it's visible to each.
            if (!getenv("SWM_NO_HUD")) {
                const char *txt = sbs
                    ? TextFormat("SBS  zoom %.2f  conv %.0f  voff %.0f  %s",
                                 zoom, conv, voff, swap ? "swapped" : "")
                    : TextFormat("FLAT  zoom %.2f  (Super+Alt+S: SBS)", zoom);
                Color c = Fade(LIME, 0.7f);
                DrawText(txt, 12, 12, 18, c);
                DrawText(txt, eyeW + 12, 12, 18, c);
            }

            rlDrawRenderBatchActive();               // flush our draws under this projection

            rlMatrixMode(RL_PROJECTION);
            rlPopMatrix();
            rlMatrixMode(RL_MODELVIEW);
            rlLoadIdentity();
        EndDrawing();
    }

    UnloadRenderTexture(rt);
    wc_input_shutdown();
    wc_shutdown();
    CloseWindow();
    return 0;
}
