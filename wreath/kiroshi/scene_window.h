// scene_window.h — draw the captured desktop as one flat, world-anchored panel.
// raylib/rlgl only (no X11). Composites each captured window at its screen
// position into a single panel rectangle.
#pragma once

#include "raylib.h"
#include "rlgl.h"
#include "window_capture.h"

// Floor grid reference height (matches scene_6dof.h eye height).
static constexpr float kPanelFloorDrop = 1.6f;

// Map a screen pixel (sx,sy) within the capture region [rx,ry, rw x rh] to a
// world point on the panel plane. Screen Y grows downward; world Y grows up.
static inline Vector3 ScreenToPanel(float sx, float sy,
                                    int rx, int ry, int rw, int rh,
                                    Vector3 center, float panelW, float panelH)
{
    float u = (sx - rx) / (float)rw;          // 0..1 left->right
    float v = (sy - ry) / (float)rh;          // 0..1 top->bottom
    return Vector3{ center.x - panelW * 0.5f + u * panelW,
                    center.y + panelH * 0.5f - v * panelH,
                    center.z };
}

// Draw the whole captured desktop as a panel `width_m` metres wide, centered at
// `center`, facing +Z (viewer at origin looking -Z). Windows are drawn in
// bottom-to-top stacking order with a slight per-window depth bias.
inline void DrawDesktopPanel(Vector3 center, float width_m)
{
    int rw = wc_screen_width(), rh = wc_screen_height();
    int rx = wc_region_x(),     ry = wc_region_y();
    if (rw <= 0 || rh <= 0) return;

    float aspect = (float)rw / (float)rh;
    float panelW = width_m;
    float panelH = width_m / aspect;

    // Colored rlgl quads need a (white) texture bound, else they sample black.
    const unsigned int white = rlGetTextureIdDefault();

    // Dark backdrop so gaps between/around windows aren't see-through.
    rlSetTexture(white);
    rlBegin(RL_QUADS);
        rlColor4ub(18, 18, 22, 255);
        float hw = panelW * 0.5f, hh = panelH * 0.5f, bz = center.z - 0.003f;
        rlTexCoord2f(0, 0); rlVertex3f(center.x - hw, center.y - hh, bz);
        rlTexCoord2f(1, 0); rlVertex3f(center.x + hw, center.y - hh, bz);
        rlTexCoord2f(1, 1); rlVertex3f(center.x + hw, center.y + hh, bz);
        rlTexCoord2f(0, 1); rlVertex3f(center.x - hw, center.y + hh, bz);
    rlEnd();

    const WCWin *wins = wc_windows();
    int n = wc_count();
    for (int i = 0; i < n; i++) {
        const WCWin &w = wins[i];
        if (w.tex == 0 || w.w <= 0 || w.h <= 0) continue;

        // Window corners in screen space -> panel plane. Nudge Z forward by
        // stacking index to honour z-order without z-fighting.
        float z = center.z + (float)(i + 1) * 0.0008f;
        Vector3 tl = ScreenToPanel(w.x,         w.y,         rx, ry, rw, rh, center, panelW, panelH);
        Vector3 tr = ScreenToPanel(w.x + w.w,   w.y,         rx, ry, rw, rh, center, panelW, panelH);
        Vector3 br = ScreenToPanel(w.x + w.w,   w.y + w.h,   rx, ry, rw, rh, center, panelW, panelH);
        Vector3 bl = ScreenToPanel(w.x,         w.y + w.h,   rx, ry, rw, rh, center, panelW, panelH);

        rlSetTexture(w.tex);
        rlBegin(RL_QUADS);
            rlColor4ub(255, 255, 255, 255);
            // V is flipped: X pixmaps are top-down. (u,v): TL=(0,0) maps to top.
            rlTexCoord2f(0, 0); rlVertex3f(tl.x, tl.y, z);   // top-left
            rlTexCoord2f(0, 1); rlVertex3f(bl.x, bl.y, z);   // bottom-left
            rlTexCoord2f(1, 1); rlVertex3f(br.x, br.y, z);   // bottom-right
            rlTexCoord2f(1, 0); rlVertex3f(tr.x, tr.y, z);   // top-right
        rlEnd();
    }
    rlSetTexture(0);

    // Cursor: draw the real pointer position on the panel (hardware cursor is
    // hidden), so the real mouse remains usable through the glasses. A green
    // crosshair over a black one for contrast on any background.
    float cu, cv;
    if (wc_pointer(&cu, &cv)) {
        Vector3 c = ScreenToPanel(rx + cu * rw, ry + cv * rh, rx, ry, rw, rh, center, panelW, panelH);
        float cz  = center.z + (float)(n + 2) * 0.0008f;
        float arm = panelW * 0.014f;    // crosshair arm length
        float th  = panelW * 0.0022f;   // arm thickness

        rlSetTexture(white);
        rlBegin(RL_QUADS);
            // black outline (slightly larger), then green on top
            for (int pass = 0; pass < 2; pass++) {
                float a = pass ? arm : arm * 1.25f;
                float t = pass ? th  : th  * 2.2f;
                float zz = pass ? cz + 0.0004f : cz;
                if (pass) rlColor4ub(60, 255, 90, 255); else rlColor4ub(0, 0, 0, 255);
                // horizontal arm
                rlTexCoord2f(0,0); rlVertex3f(c.x - a, c.y - t, zz);
                rlTexCoord2f(1,0); rlVertex3f(c.x + a, c.y - t, zz);
                rlTexCoord2f(1,1); rlVertex3f(c.x + a, c.y + t, zz);
                rlTexCoord2f(0,1); rlVertex3f(c.x - a, c.y + t, zz);
                // vertical arm
                rlTexCoord2f(0,0); rlVertex3f(c.x - t, c.y - a, zz);
                rlTexCoord2f(1,0); rlVertex3f(c.x + t, c.y - a, zz);
                rlTexCoord2f(1,1); rlVertex3f(c.x + t, c.y + a, zz);
                rlTexCoord2f(0,1); rlVertex3f(c.x - t, c.y + a, zz);
            }
        rlEnd();
    }
    rlSetTexture(0);
}
