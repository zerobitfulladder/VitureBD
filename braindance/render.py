"""render.py — moderngl rendering: composite captured windows into a work-area
texture, then present to the glasses either FLAT (2D, both eyes identical) or
DEPTH (mesh-warp DIBR per eye). Zoom/pan apply to both modes; disparity is the
3D strength (FLAT == DEPTH with disparity 0).

Orientation convention: the desktop's top maps to texture v=1 everywhere (work
RT, depth texture), so color and depth align with no per-stage Y flips. The mesh
maps uv.y=1 -> screen top.
"""
import numpy as np
import moderngl

from . import panel
from .panel import Panel

# how far the settings panel floats in front of the scene (NDC disparity)
PANEL_DISP = 0.018

# --- shaders ---------------------------------------------------------------

COMPOSITE_VS = """
#version 330
in vec2 in_uv;
uniform vec4 rect;     // window rect in region px (x0,y0,x1,y1)
uniform vec2 region;   // (rw, rh)
out vec2 v_uv;
void main() {
    vec2 px = mix(rect.xy, rect.zw, in_uv);
    vec2 ndc = px / region * 2.0 - 1.0;
    ndc.y = -ndc.y;                       // py=0 (top) -> +1 ; RT v=1 = desktop top
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_uv = in_uv;
}
"""
COMPOSITE_FS = """
#version 330
in vec2 v_uv;
uniform sampler2D win_tex;
out vec4 f;
void main() { f = texture(win_tex, v_uv); }
"""

# Full-screen blit (work RT -> downscaled depth-source RT)
BLIT_VS = """
#version 330
in vec2 in_uv;
out vec2 v_uv;
void main() { gl_Position = vec4(in_uv * 2.0 - 1.0, 0.0, 1.0); v_uv = in_uv; }
"""
BLIT_FS = """
#version 330
in vec2 v_uv;
uniform sampler2D src;
out vec4 f;
void main() { f = texture(src, v_uv); }
"""

# Per-eye present (mesh warp). disp=0 -> FLAT. Each vertex slides sideways by its
# depth; z puts nearer vertices in front, so where the warp folds a near edge
# over the background the near edge wins whatever order the rows are drawn in.
WARP_VS = """
#version 330
in vec2 in_uv;
uniform sampler2D depth_tex;
uniform float disp;      // 3D strength: disparity scale (NDC x)
uniform float vshift;    // SCREEN DEPTH: constant per-eye plane slide (NDC); floats the
                         // whole scene in/out, works in FLAT (disp 0) and DEPTH alike
uniform float eye_sign;  // +1 / -1
uniform float zoom;
uniform vec2  pan;       // NDC
uniform float climit;    // comfort clamp on |disparity| (NDC); <=0 = off
out vec2 v_uv;
out float v_d;
void main() {
    float d = texture(depth_tex, in_uv).r;
    vec2 p = in_uv * 2.0 - 1.0;          // uv.y=1 -> +1 (top)
    p *= zoom;
    p += pan;
    float dval = disp * (d - 0.5);       // linear: zero-parallax at mid-depth
    if (climit > 0.0) dval = clamp(dval, -climit, climit);   // comfort clamp (3D pop only)
    p.x += eye_sign * (dval + vshift);   // + SCREEN DEPTH plane slide
    gl_Position = vec4(p, 0.999 * (1.0 - 2.0 * clamp(d, 0.0, 1.0)), 1.0);
    v_uv = in_uv;
    v_d = d;
}
"""
# Besides the color, the mesh writes how stretched each pixel is: where depth
# rises steeply in the direction this eye shifts near things, the triangles
# there are drawn many times wider than they are in the desktop. Across a depth
# edge that stretch is the background the near object uncovered, painted with
# the edge's own colours (the streaks). EDGE_FILL_FS replaces it.
WARP_FS = """
#version 330
in vec2 v_uv;
in float v_d;
uniform sampler2D color_tex;
uniform sampler2D depth_tex;
uniform float show_depth;     // 1 -> draw the depth map (grayscale) instead of color
uniform float disp;           // as in WARP_VS
uniform float eye_sign;
uniform float zoom;
uniform float climit;
uniform float dtexel;         // 1 / depth width
uniform float src_px;         // target px per uv (eye width x zoom)
layout(location = 0) out vec4 f;
layout(location = 1) out vec4 m;  // stretch, 1 if the background lies left, unwarped x (px)
float shift(float d) {
    float s = disp * (d - 0.5);
    return climit > 0.0 ? clamp(s, -climit, climit) : s;
}
void main() {
    f = (show_depth > 0.5) ? vec4(vec3(v_d), 1.0) : texture(color_tex, v_uv);
    float dl = texture(depth_tex, v_uv - vec2(dtexel, 0.0)).r;
    float dr = texture(depth_tex, v_uv + vec2(dtexel, 0.0)).r;
    float stretch = 1.0 + eye_sign * (shift(dr) - shift(dl)) / (4.0 * zoom * dtexel);
    m = vec4(stretch, dl < dr ? 1.0 : 0.0, v_uv.x * src_px, 1.0);
}
"""

# Disocclusion fill. Along a row, a run of stretched pixels is either a steep
# surface (drawn wider in this eye, as it should be) or a depth edge, where the
# run is wider than the desktop pixels it shows by the background the near
# object uncovered. A run that uncovers `cliff_px` or more is an edge: it gets
# the background beside it mirrored in, and so do `halo` px of that background,
# which carry the near object's blurred rim (left behind at background depth,
# it would outline the fill). The fill blends into the near object over
# `feather` px. Output: rgb = fill colour, a = its weight (0 = keep the mesh).
EDGE_FILL_FS = """
#version 330
in vec2 v_uv;
uniform sampler2D msk;        // WARP_FS's stretch map
uniform sampler2D col;        // WARP_FS's color
uniform vec2 texel;
uniform float cliff_px;
uniform float feather;
uniform int halo;
out vec4 f;
const float RUN = 1.5;        // stretch from which a pixel belongs to a run
const int MAXW = 400;         // the widest run looked at (px)
vec4 at(int i, float dir) { return texture(msk, v_uv + vec2(dir * float(i) * texel.x, 0.0)); }
// The run through pixel `start` (in steps of dir): kb, the first pixel past it
// toward the background; ko, toward the near object. Returns how much of a
// cliff it is (0..1) from the px it uncovers.
float cliff(int start, float dir, out int kb, out int ko) {
    kb = 0; ko = 0;
    for (int i = start; i <= start + MAXW; i++) if (at(i, dir).r <= RUN) { kb = i; break; }
    for (int i = start; i >= start - MAXW; i--) if (at(i, dir).r <= RUN) { ko = i; break; }
    if (kb == 0 || ko == 0) return 0.0;
    float uncovered = float(kb - ko) - abs(at(kb, dir).b - at(ko, dir).b);
    return smoothstep(0.5 * cliff_px, 1.5 * cliff_px, uncovered);
}
void main() {
    vec4 mk = texture(msk, v_uv);
    float dir = 0.0, w = 0.0;
    int c0 = 0, kb, ko;                   // c0: px to the clean background, along dir
    if (mk.r > RUN) {
        dir = mk.g > 0.5 ? -1.0 : 1.0;
        float wc = cliff(0, dir, kb, ko);
        if (wc > 0.0) { c0 = kb + halo; w = wc * smoothstep(0.0, feather, float(-ko)); }
    } else {
        // in the halo: the nearest run within `halo` px on either side, if its
        // background lies on our side
        for (int side = -1; side <= 1 && c0 == 0; side += 2) {
            for (int j = 1; j <= halo; j++) {
                vec4 a = at(j, float(side));
                if (a.r <= RUN) continue;
                if ((side < 0) == (a.g < 0.5)) {
                    float wc = cliff(-j, -float(side), kb, ko);
                    if (wc > 0.0) { dir = -float(side); c0 = halo - j + 1; w = wc; }
                }
                break;
            }
        }
    }
    if (c0 == 0) { f = vec4(0.0); return; }
    vec2 stp = vec2(dir * texel.x, 0.0);
    vec2 s = v_uv + float(2 * c0) * stp;  // mirror about the clean edge
    if (texture(msk, s).r > RUN) s = v_uv + float(c0) * stp;
    f = vec4(texture(col, s).rgb, w);
}
"""

# Packs one eye, shrunk to the inpainting network's size (inpaint.py), into its
# half of the network's input: RGB, and as alpha the mask of what to paint, the
# filled pixels grown over the same 13 x 3 window EDGE_MIX_FS blends over. Rows
# go top first (CUDA reads row 0 first), and the left eye is mirrored: the
# network expects the holes on the right of near things, as the right eye has.
EDGE_PACK_FS = """
#version 330
in vec2 v_uv;
uniform sampler2D col;        // the eye's mesh colour
uniform sampler2D fil;        // its EDGE_FILL_FS output
uniform float mirror;         // 1 for the left eye
uniform vec2 texel;           // one eye-target px, in uv
uniform vec2 step;            // one packed px, in uv
out vec4 f;
void main() {
    vec2 uv = vec2(mirror > 0.5 ? 1.0 - v_uv.x : v_uv.x, 1.0 - v_uv.y);
    vec3 c = vec3(0.0);
    for (int i = 0; i < 4; i++)
        c += texture(col, uv + (vec2(float(i & 1), float(i >> 1)) - 0.5) * 0.5 * step).rgb;
    float m = 0.0;
    for (int dy = -6; dy <= 6; dy += 2)
        for (int dx = -1; dx <= 1; dx++)
            m = max(m, texture(fil, uv + vec2(float(dx), float(dy)) * texel).a);
    f = vec4(0.25 * c, m > 0.0 ? 1.0 : 0.0);
}
"""

# Blends the fill into the mesh. The fill is decided row by row, so its weight
# is averaged over a tall window (13 x 3) of the filled pixels, and so is the
# mirror fill's colour: rows that disagree fade into each other instead of
# stepping. The network's picture comes in whole, upsampled.
EDGE_MIX_FS = """
#version 330
in vec2 v_uv;
uniform sampler2D col;
uniform sampler2D fil;        // EDGE_FILL_FS's output
uniform sampler2D net;        // the network's picture of both eyes, packed as EDGE_PACK_FS
uniform vec2 texel;
uniform float source;         // 0 the mirror fill, 1 the network's, 2 magenta (FILL DEBUG)
uniform vec2 net_x;           // this eye's u in `net`: net_x.x + net_x.y * u
out vec4 f;
void main() {
    vec3 c = texture(col, v_uv).rgb;
    vec3 s = vec3(0.0);
    float n = 0.0, wsum = 0.0;
    for (int dy = -6; dy <= 6; dy++)
        for (int dx = -1; dx <= 1; dx++) {
            vec4 q = texture(fil, v_uv + vec2(float(dx), float(dy)) * texel);
            if (q.a > 0.0) { s += q.rgb; n += 1.0; wsum += q.a; }
        }
    if (n == 0.0) { f = vec4(c, 1.0); return; }
    float w = max(texture(fil, v_uv).a, wsum / 39.0);
    vec3 fc = source > 1.5 ? vec3(1.0, 0.0, 0.6)
            : source > 0.5 ? texture(net, vec2(net_x.x + net_x.y * v_uv.x, 1.0 - v_uv.y)).rgb
            : s / n;
    f = vec4(mix(c, fc, w), 1.0);
}
"""

# The cursor: a small classic arrow pointing up-left, its tip (the hotspot) at
# the local origin, the left edge straight down, the diagonal back up-right.
CURSOR_TRI = np.array([0.0, 0.0,  0.0, -1.4,  0.95, -0.95], dtype="f4")
CURSOR_VS = """
#version 330
in vec2 in_p;          // unit quad [-1,1]
uniform vec2 center;   // NDC
uniform vec2 halfsz;   // NDC
void main() { gl_Position = vec4(center + in_p * halfsz, 0.0, 1.0); }
"""
CURSOR_FS = """
#version 330
uniform vec3 col;
out vec4 f;
void main() { f = vec4(col, 1.0); }
"""

# SSAA resolve: 4-tap box downsample of the supersampled present buffer. Each tap
# is itself bilinear, so this averages the warped silhouette across sub-pixels —
# smooths the depth-edge "ladder" that finer tessellation only shrinks.
RESOLVE_FS = """
#version 330
in vec2 v_uv;
uniform sampler2D src;
uniform vec2 texel;     // 1 / supersampled size
out vec4 f;
void main() {
    vec2 o = texel * 0.5;
    vec3 c = texture(src, v_uv + vec2(-o.x, -o.y)).rgb
           + texture(src, v_uv + vec2( o.x, -o.y)).rgb
           + texture(src, v_uv + vec2(-o.x,  o.y)).rgb
           + texture(src, v_uv + vec2( o.x,  o.y)).rgb;
    f = vec4(c * 0.25, 1.0);
}
"""

# Edge-aware depth UPSAMPLE: a COLOR-guided joint-bilateral filter run at HIGH
# output resolution. For each output texel, average low-res depth samples weighted
# by spatial distance AND color similarity to the center (full-res color guide) —
# so the soft, low-res DA2 edge snaps onto the real silhouette at high res (kills
# the laddery edges). dtexel = OUTPUT texel (offsets walk the full-res guide);
# sigma_c small = tighter edge-snapping.
SHARP_FS = """
#version 330
in vec2 v_uv;
uniform sampler2D depth_tex;   // unit 1 (low-res, bilinear)
uniform sampler2D color_tex;   // unit 0 (full-res guide)
uniform vec2 dtexel;           // gather stride = 1 / DEPTH size (low-res)
uniform float reach;           // stride multiplier: widen search w/o more taps
uniform float sigma_c;         // color sigma (normalized)
uniform float sigma_d;         // depth-range sigma (normalized; large => off)
out vec4 f;
const int R = 4;
void main() {
    vec3  c0 = texture(color_tex, v_uv).rgb;  // this output pixel's full-res color
    float d0 = texture(depth_tex, v_uv).r;    // center depth (for range term)
    float sum = 0.0, wsum = 0.0;
    for (int dy = -R; dy <= R; dy++) {
        for (int dx = -R; dx <= R; dx++) {
            vec2  off = vec2(dx, dy) * dtexel * reach;
            float d = texture(depth_tex, v_uv + off).r;
            vec3  c = texture(color_tex, v_uv + off).rgb;
            float sp = float(dx * dx + dy * dy);
            vec3  dc = c - c0;
            float dd = d - d0;
            float w = exp(-sp / 14.0)
                    * exp(-dot(dc, dc) / (2.0 * sigma_c * sigma_c))
                    * exp(-(dd * dd)   / (2.0 * sigma_d * sigma_d));
            sum += w * d;
            wsum += w;
        }
    }
    f = vec4(sum / max(wsum, 1e-5), 0.0, 0.0, 1.0);
}
"""

# --- fast guided filter (He et al.), luma guide ----------------------------
# An alternative to JBU whose heavy work runs at the low depth resolution (cheap),
# not full-res like JBU — so it's faster at typical depth res, and its cost is
# independent of the OUTPUT res. (Box is separable but naive => O(radius), not a
# true O(1) SAT box.) Only the final apply is full-res. Each output carries a
# local linear model q = a*I + b, so flat regions (low var) just smooth (no
# texture-copy) while edges (high var) snap to the guide. eps = smoothing knob.
#
# Pipeline (depth res, except apply): moments -> boxH -> boxV -> coeff ->
# boxH -> boxV -> apply@full-res.

# P1: build per-pixel moments. I = luma(guide), p = depth.
GF_MOM_FS = """
#version 330
in vec2 v_uv;
uniform sampler2D color_tex;   // unit 0, full-res guide (bilinear -> downsample)
uniform sampler2D depth_tex;   // unit 1, low-res depth
out vec4 f;
const vec3 LUMA = vec3(0.299, 0.587, 0.114);
void main() {
    float I = dot(texture(color_tex, v_uv).rgb, LUMA);
    float p = texture(depth_tex, v_uv).r;
    f = vec4(I, p, I * p, I * I);   // (mean targets: I, p, Ip, II)
}
"""

# Separable box filter (run H then V). Dynamic radius w/ a fixed loop bound.
BOX_FS = """
#version 330
in vec2 v_uv;
uniform sampler2D src;
uniform vec2 dir;       // (1/w,0) horizontal or (0,1/h) vertical
uniform int radius;
out vec4 f;
const int MAXR = 16;
void main() {
    vec4 s = vec4(0.0); float n = 0.0;
    for (int i = -MAXR; i <= MAXR; i++) {
        if (i < -radius || i > radius) continue;
        s += texture(src, v_uv + dir * float(i));
        n += 1.0;
    }
    f = s / n;
}
"""

# P4: per-pixel linear coefficients from the box-averaged moments.
GF_COEFF_FS = """
#version 330
in vec2 v_uv;
uniform sampler2D mean_tex;    // (mean_I, mean_p, mean_Ip, mean_II)
uniform float eps;
out vec4 f;
void main() {
    vec4  m = texture(mean_tex, v_uv);
    float varI  = m.w - m.x * m.x;
    float covIp = m.z - m.x * m.y;
    float a = covIp / (varI + eps);
    float b = m.y - a * m.x;
    f = vec4(a, b, 0.0, 0.0);
}
"""

# P7: apply the (box-averaged) linear model at full res. This IS the upsample.
GF_APPLY_FS = """
#version 330
in vec2 v_uv;
uniform sampler2D ab_tex;      // unit 0, low-res (a,b) -> bilinear upsample
uniform sampler2D color_tex;   // unit 1, full-res guide
out vec4 f;
const vec3 LUMA = vec3(0.299, 0.587, 0.114);
void main() {
    vec2  ab = texture(ab_tex, v_uv).rg;
    float I  = dot(texture(color_tex, v_uv).rgb, LUMA);
    f = vec4(clamp(ab.x * I + ab.y, 0.0, 1.0), 0.0, 0.0, 1.0);
}
"""

# Depth debug: draw the depth map straight as grayscale (near=white, far=black).
DEPTHVIEW_FS = """
#version 330
in vec2 v_uv;
uniform sampler2D depth_tex;
out vec4 f;
void main() { float d = texture(depth_tex, v_uv).r; f = vec4(vec3(d), 1.0); }
"""

# Per-eye present, learned backward warp (stereo.py): each pixel of an eye is the
# weighted blend of the desktop sampled at two horizontal offsets the network
# predicted for it. Zoom, pan and SCREEN DEPTH map the eye's pixel back to the
# scene first, as the mesh warp maps the scene forward.
MLBW_FS = """
#version 330
in vec2 v_uv;                 // 0..1 over this eye's viewport, v=1 = top
uniform sampler2D color_tex;  // the desktop, v=1 = top
uniform sampler2D depth_tex;
uniform sampler2D flow_tex;   // this eye's (d0, d1, w0, w1); offsets in network px
uniform float eye_sign;       // +1 left eye: sample at x + d; -1 right eye: x - d
uniform float uv_scale;       // network px -> uv
uniform float zoom;
uniform vec2  pan;            // NDC
uniform float vshift;         // SCREEN DEPTH: constant per-eye plane slide (NDC)
uniform float climit;         // comfort clamp on |offset| (uv); <= 0 = off
uniform float show_depth;     // 1 -> warp the depth map (grayscale) instead of color
out vec4 f;
vec3 src(vec2 uv) {
    return show_depth > 0.5 ? vec3(texture(depth_tex, uv).r) : texture(color_tex, uv).rgb;
}
void main() {
    vec2 p = v_uv * 2.0 - 1.0;
    p.x -= eye_sign * vshift;
    vec2 s = (p - pan) / zoom * 0.5 + 0.5;
    if (s.x < 0.0 || s.y < 0.0 || s.x > 1.0 || s.y > 1.0) { f = vec4(0.0, 0.0, 0.0, 1.0); return; }
    vec4 fl = texture(flow_tex, s);
    vec2 d = fl.xy * (uv_scale * eye_sign);
    if (climit > 0.0) d = clamp(d, -climit, climit);
    f = vec4(fl.z * src(vec2(s.x + d.x, s.y)) + fl.w * src(vec2(s.x + d.y, s.y)), 1.0);
}
"""


def _grid(gx, gy):
    xs = np.linspace(0, 1, gx + 1, dtype="f4")
    ys = np.linspace(0, 1, gy + 1, dtype="f4")
    uv = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2).astype("f4")
    idx, stride = [], gx + 1
    for j in range(gy):
        for i in range(gx):
            a = j * stride + i; b = a + 1; c = a + stride; d = c + 1
            idx += [a, b, c, b, d, c]
    return uv, np.array(idx, dtype="i4")


class Renderer:
    def __init__(self, ctx, pw, ph, eye_w, region, grid=256, depth_h=392, ss=1.0):
        self.ctx = ctx
        self.screen = ctx.screen        # where draw() presents; a framebuffer offscreen
        self.pw, self.ph, self.eye_w = pw, ph, eye_w
        self.rw, self.rh = region
        self.eye_h = ph
        self.ss = max(1.0, float(ss))   # present supersampling factor (1.0 = off)

        # work-area RT (composited desktop), v=1 = top
        self.work = ctx.framebuffer(color_attachments=[ctx.texture((self.rw, self.rh), 3)])
        self.work.color_attachments[0].filter = (moderngl.LINEAR, moderngl.LINEAR)
        # clamp to edge: the mlbw warp samples past the borders (grid_sample "border")
        self.work.color_attachments[0].repeat_x = False
        self.work.color_attachments[0].repeat_y = False

        # downscaled RT for depth-model readback
        dw = max(14, int(round(depth_h * self.rw / self.rh / 14)) * 14)
        self.dw, self.dh = dw, depth_h
        self.dsrc = ctx.framebuffer(color_attachments=[ctx.texture((dw, depth_h), 3)])
        self.dsrc.color_attachments[0].filter = (moderngl.LINEAR, moderngl.LINEAR)
        # double-buffered PBOs for async readback (no glReadPixels stall)
        self._pbos = [ctx.buffer(reserve=dw * depth_h * 3) for _ in range(2)]
        self._pbo_i = 0
        self._pbo_primed = False

        # depth and per-eye stereo offsets, at the depth pipeline's size (attach_depth)
        self.depth_tex = self._float_tex((dw, depth_h), 1)
        self.depth_tex.write(np.full((depth_h, dw), 0.5, dtype="f4").tobytes())
        self.flow_tex = [None, None]
        self.has_flow = False         # the flow textures hold offsets for the current depth
        self.flow_uv = 0.0            # network px -> uv for those offsets

        # GL<->CUDA interop for the depth pipeline (attach_depth). None when off.
        self.idsrc = None
        self._ireader = self._iwriter = None
        self._fwriters = [None, None]

        # edge-aware depth UPSAMPLE: color-guided joint-bilateral into dref_tex at
        # HIGH resolution (not the model's low res), so the warp samples crisp,
        # color-aligned depth edges instead of the laddery low-res ones.
        self._refh = self.rh                           # refine at full region res (1080)
        self._refw = max(2, int(round(self._refh * self.rw / self.rh)))
        self.dref_tex = None
        self.dref_fbo = None
        self._ensure_dref()

        # programs / geometry
        unit = np.array([0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1], dtype="f4")
        self.comp = ctx.program(vertex_shader=COMPOSITE_VS, fragment_shader=COMPOSITE_FS)
        self.comp["region"].value = (float(self.rw), float(self.rh))
        self.comp_vao = ctx.vertex_array(self.comp, [(ctx.buffer(unit.tobytes()), "2f", "in_uv")])

        self.blit = ctx.program(vertex_shader=BLIT_VS, fragment_shader=BLIT_FS)
        self.blit_vao = ctx.vertex_array(self.blit, [(ctx.buffer(unit.tobytes()), "2f", "in_uv")])

        self.sharp = ctx.program(vertex_shader=BLIT_VS, fragment_shader=SHARP_FS)
        self.sharp["color_tex"] = 0
        self.sharp["depth_tex"] = 1
        self.sharp_vao = ctx.vertex_array(self.sharp, [(ctx.buffer(unit.tobytes()), "2f", "in_uv")])

        # guided-filter programs (luma guide) + ping-pong RTs at depth res
        self.gf_mom = ctx.program(vertex_shader=BLIT_VS, fragment_shader=GF_MOM_FS)
        self.gf_mom["color_tex"] = 0
        self.gf_mom["depth_tex"] = 1
        self.gf_mom_vao = ctx.vertex_array(self.gf_mom, [(ctx.buffer(unit.tobytes()), "2f", "in_uv")])
        self.box = ctx.program(vertex_shader=BLIT_VS, fragment_shader=BOX_FS)
        self.box["src"] = 0
        self.box_vao = ctx.vertex_array(self.box, [(ctx.buffer(unit.tobytes()), "2f", "in_uv")])
        self.gf_coeff = ctx.program(vertex_shader=BLIT_VS, fragment_shader=GF_COEFF_FS)
        self.gf_coeff["mean_tex"] = 0
        self.gf_coeff_vao = ctx.vertex_array(self.gf_coeff, [(ctx.buffer(unit.tobytes()), "2f", "in_uv")])
        self.gf_apply = ctx.program(vertex_shader=BLIT_VS, fragment_shader=GF_APPLY_FS)
        self.gf_apply["ab_tex"] = 0
        self.gf_apply["color_tex"] = 1
        self.gf_apply_vao = ctx.vertex_array(self.gf_apply, [(ctx.buffer(unit.tobytes()), "2f", "in_uv")])
        self.gf_a_tex = self.gf_b_tex = None
        self.gf_a_fbo = self.gf_b_fbo = None

        self.depthview = ctx.program(vertex_shader=BLIT_VS, fragment_shader=DEPTHVIEW_FS)
        self.depthview["depth_tex"] = 0
        self.depthview_vao = ctx.vertex_array(
            self.depthview, [(ctx.buffer(unit.tobytes()), "2f", "in_uv")])
        self.mlbw = ctx.program(vertex_shader=BLIT_VS, fragment_shader=MLBW_FS)
        self.mlbw["color_tex"] = 0
        self.mlbw["depth_tex"] = 1
        self.mlbw["flow_tex"] = 2
        self.mlbw_vao = ctx.vertex_array(self.mlbw, [(ctx.buffer(unit.tobytes()), "2f", "in_uv")])

        gy = max(1, grid * self.rh // self.rw)
        uv, idx = _grid(grid, gy)
        uvbuf, idxbuf = ctx.buffer(uv.tobytes()), ctx.buffer(idx.tobytes())
        self.warp = ctx.program(vertex_shader=WARP_VS, fragment_shader=WARP_FS)
        self.warp["color_tex"] = 0
        self.warp["depth_tex"] = 1
        self.warp_vao = ctx.vertex_array(self.warp, [(uvbuf, "2f", "in_uv")], idxbuf)
        # disocclusion fill: each eye's mesh goes to its own targets (eye_fbo), and
        # the fill and the blend bring it to the screen
        self.edge_fill = ctx.program(vertex_shader=BLIT_VS, fragment_shader=EDGE_FILL_FS)
        self.edge_fill["msk"] = 0
        self.edge_fill["col"] = 1
        self.edge_fill_vao = ctx.vertex_array(
            self.edge_fill, [(ctx.buffer(unit.tobytes()), "2f", "in_uv")])
        self.edge_mix = ctx.program(vertex_shader=BLIT_VS, fragment_shader=EDGE_MIX_FS)
        self.edge_mix["col"] = 0
        self.edge_mix["fil"] = 1
        self.edge_mix["net"] = 2
        self.edge_mix_vao = ctx.vertex_array(
            self.edge_mix, [(ctx.buffer(unit.tobytes()), "2f", "in_uv")])
        self.eye_fbo = self.fill_fbo = None     # [left, right]
        # the network fill: both eyes packed for it, and its picture (_ensure_net)
        self.edge_pack = ctx.program(vertex_shader=BLIT_VS, fragment_shader=EDGE_PACK_FS)
        self.edge_pack["col"] = 0
        self.edge_pack["fil"] = 1
        self.edge_pack_vao = ctx.vertex_array(
            self.edge_pack, [(ctx.buffer(unit.tobytes()), "2f", "in_uv")])
        self.pack_fbo = self.net_tex = None
        self._preader = self._nwriter = None

        # the cursor (CURSOR_TRI), at a constant on-screen size, with its own
        # disparity to float above
        self.cursor = ctx.program(vertex_shader=CURSOR_VS, fragment_shader=CURSOR_FS)
        self.cursor_vao = ctx.vertex_array(self.cursor, [(ctx.buffer(CURSOR_TRI.tobytes()), "2f", "in_p")])
        self.aspect = eye_w / ph

        # settings panel (opened with the mouse's side button), drawn per eye by draw_panel
        self.panel = Panel(ctx)

        # SSAA present buffer: render the warp + cursor at ss x panel res into an
        # offscreen RT, then box-downsample to the screen (see RESOLVE_FS). The
        # resolve program is always built so SSAA can be toggled live (set_ss).
        self.resolve = ctx.program(vertex_shader=BLIT_VS, fragment_shader=RESOLVE_FS)
        self.resolve["src"] = 0
        self.resolve_vao = ctx.vertex_array(
            self.resolve, [(ctx.buffer(unit.tobytes()), "2f", "in_uv")])
        self.ss_tex = None
        self.ss_fbo = None
        self._alloc_ss()

    def _alloc_ss(self):
        """(Re)allocate the supersample RT for the current self.ss (None if off)."""
        if self.ss_fbo is not None:
            self.ss_fbo.release()
            self.ss_tex.release()
            self.ss_fbo = self.ss_tex = None
        if self.ss > 1.0:
            sw, sh = int(round(self.pw * self.ss)), int(round(self.ph * self.ss))
            self.ss_tex = self.ctx.texture((sw, sh), 3)
            self.ss_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.ss_fbo = self.ctx.framebuffer(color_attachments=[self.ss_tex])
            self.resolve["texel"].value = (1.0 / sw, 1.0 / sh)

    def set_ss(self, factor):
        """Change the present supersampling factor at runtime (rebuilds the RT)."""
        factor = max(1.0, float(factor))
        if factor == self.ss:
            return
        self.ss = factor
        self._alloc_ss()

    def _ensure_eye_rt(self, ew, eh):
        """Each eye's mesh targets (color, stretch map, depth buffer) and fill
        target, at the present size. Clamped, so the fill's row walks stop at
        the eye's edge; the fill's taps land on pixel centres, so the color's
        LINEAR (for EDGE_PACK_FS) reads the same whole pixels there."""
        if self.eye_fbo is not None and self.eye_fbo[0].size == (ew, eh):
            return
        for fbo in (*(self.eye_fbo or ()), *(self.fill_fbo or ())):
            for t in (*fbo.color_attachments, fbo.depth_attachment):
                if t is not None:
                    t.release()
            fbo.release()
        self.eye_fbo, self.fill_fbo = [], []
        for _ in range(2):
            col = self.ctx.texture((ew, eh), 4)
            msk = self.ctx.texture((ew, eh), 4, dtype="f4")
            fil = self.ctx.texture((ew, eh), 4)
            for t in (msk, fil):
                t.filter = (moderngl.NEAREST, moderngl.NEAREST)
            for t in (col, msk, fil):
                t.repeat_x = t.repeat_y = False
            self.eye_fbo.append(self.ctx.framebuffer(
                color_attachments=[col, msk], depth_attachment=self.ctx.depth_renderbuffer((ew, eh))))
            self.fill_fbo.append(self.ctx.framebuffer(color_attachments=[fil]))

    def _ensure_net(self, size):
        """The network fill's packed input (RGBA8, read by CUDA) and its picture
        (RGBA32F, written by CUDA), both eyes side by side at the network's
        per-eye `size` (h, w)."""
        h, w = size
        if self.pack_fbo is not None and self.pack_fbo.size == (2 * w, h):
            return
        self._release_net()
        from .glcuda import GLTexReader, GLTexWriter
        self.pack_fbo = self.ctx.framebuffer(color_attachments=[self.ctx.texture((2 * w, h), 4)])
        self.net_tex = self._float_tex((2 * w, h), 4)
        self._preader = GLTexReader(self.pack_fbo.color_attachments[0].glo, 2 * w, h)
        self._nwriter = GLTexWriter(self.net_tex.glo, 2 * w, h, channels=4)

    def _release_net(self):
        for o in (self._preader, self._nwriter):
            if o is not None:
                o.close()
        self._preader = self._nwriter = None
        if self.pack_fbo is not None:
            self.pack_fbo.color_attachments[0].release()
            self.pack_fbo.release()
            self.net_tex.release()
            self.pack_fbo = self.net_tex = None

    def _paint(self, inpaint, ew, eh):
        """Run the network fill on both eyes' fills: pack them, paint, and put
        its picture in net_tex. False when nothing needs painting."""
        self._ensure_net(inpaint.size)
        h, w = inpaint.size
        self.edge_pack["texel"].value = (1.0 / ew, 1.0 / eh)
        self.edge_pack["step"].value = (1.0 / w, 1.0 / h)
        self.pack_fbo.use()
        for half in range(2):
            self.ctx.viewport = (half * w, 0, w, h)
            self.eye_fbo[half].color_attachments[0].use(0)
            self.fill_fbo[half].color_attachments[0].use(1)
            self.edge_pack["mirror"].value = 1.0 if half == 0 else 0.0
            self.edge_pack_vao.render(moderngl.TRIANGLES)
        packed = self._preader.read()
        if not bool(packed[..., 3].any()):
            return False
        self._nwriter.write(inpaint(packed))
        return True

    # --- desktop compositing -------------------------------------------------
    def composite(self, windows):
        self.work.use()
        self.ctx.clear(0.07, 0.07, 0.09)
        for tex, x, y, w, h in windows:
            # A fullscreen video can escape the 1920 box to the whole 3840 panel.
            # We hold the full window pixmap, so show its CENTER (region-sized) —
            # where a 16:9 video sits (pillarboxed) — instead of only its left half.
            if w > self.rw:
                x = (self.rw - w) * 0.5
            if h > self.rh:
                y = (self.rh - h) * 0.5
            tex.use(0)
            self.comp["rect"].value = (float(x), float(y), float(x + w), float(y + h))
            self.comp_vao.render(moderngl.TRIANGLES)

    def depth_frame(self, use_pbo=True):
        """Downscale the work RT and read it back as an RGB uint8 frame (HxWx3)
        for the depth model. Top row = desktop top, matching the depth upload.

        use_pbo=True (async): double-buffered PBO, no GPU stall, returns the
        PREVIOUS frame (1 cycle old) — for the threaded worker.
        use_pbo=False (sync): synchronous read of the CURRENT frame so depth is
        matched exactly (no 1-frame desync — clean during scroll)."""
        self.dsrc.use()
        self.ctx.clear()
        self.work.color_attachments[0].use(0)
        self.blit_vao.render(moderngl.TRIANGLES)

        if not use_pbo:
            data = self.dsrc.read(components=3, dtype="f1")   # current frame
            arr = np.frombuffer(data, dtype="u1").reshape(self.dh, self.dw, 3)
            return np.ascontiguousarray(arr[::-1])

        i = self._pbo_i
        self.dsrc.read_into(self._pbos[i], components=3, dtype="f1")  # async into PBO i
        j = 1 - i
        self._pbo_i = j
        if not self._pbo_primed:
            self._pbo_primed = True
            return None
        data = self._pbos[j].read()   # PBO written last call — transfer already done
        arr = np.frombuffer(data, dtype="u1").reshape(self.dh, self.dw, 3)
        return np.ascontiguousarray(arr[::-1])   # fbo is bottom-up -> flip to top-down

    def _ensure_dref(self):
        """Create the high-res refined-depth RT (fixed size, independent of the
        model's depth resolution — it's the upsample target)."""
        wh = (self._refw, self._refh)
        if self.dref_tex is not None and self.dref_tex.size == wh:
            return
        if self.dref_fbo is not None:
            self.dref_fbo.release()
            self.dref_tex.release()
        self.dref_tex = self.ctx.texture(wh, 1, dtype="f4")
        self.dref_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.dref_fbo = self.ctx.framebuffer(color_attachments=[self.dref_tex])

    def _ensure_gf(self, size):
        """(Re)allocate the guided-filter ping-pong RTs (RGBA32F at depth res)."""
        if self.gf_a_tex is not None and self.gf_a_tex.size == size:
            return
        for t in (self.gf_a_fbo, self.gf_a_tex, self.gf_b_fbo, self.gf_b_tex):
            if t is not None:
                t.release()
        self.gf_a_tex = self.ctx.texture(size, 4, dtype="f4")
        self.gf_b_tex = self.ctx.texture(size, 4, dtype="f4")
        for t in (self.gf_a_tex, self.gf_b_tex):
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.gf_a_fbo = self.ctx.framebuffer(color_attachments=[self.gf_a_tex])
        self.gf_b_fbo = self.ctx.framebuffer(color_attachments=[self.gf_b_tex])

    def _guided(self, eps, radius):
        """Fast guided filter (luma guide): refine low-res depth into dref_tex at
        full res. Moment passes run at depth res; only the apply is full-res."""
        ctx = self.ctx
        dw, dh = self.depth_tex.size
        self._ensure_gf((dw, dh))
        hx, hy = (1.0 / dw, 0.0), (0.0, 1.0 / dh)

        def box(src_tex, dst_fbo, dirv):                # framebuffer.use() sets viewport
            src_tex.use(0); dst_fbo.use()
            self.box["dir"].value = dirv
            self.box["radius"].value = int(radius)
            self.box_vao.render(moderngl.TRIANGLES)

        self.work.color_attachments[0].use(0); self.depth_tex.use(1)
        self.gf_a_fbo.use()                              # P1 moments -> A
        self.gf_mom_vao.render(moderngl.TRIANGLES)
        box(self.gf_a_tex, self.gf_b_fbo, hx)           # P2 boxH -> B
        box(self.gf_b_tex, self.gf_a_fbo, hy)           # P3 boxV -> A (= means)
        self.gf_a_tex.use(0); self.gf_b_fbo.use()       # P4 coeff -> B (= a,b)
        self.gf_coeff["eps"].value = float(eps)
        self.gf_coeff_vao.render(moderngl.TRIANGLES)
        box(self.gf_b_tex, self.gf_a_fbo, hx)           # P5 boxH -> A
        box(self.gf_a_tex, self.gf_b_fbo, hy)           # P6 boxV -> B (= mean a,b)
        self._ensure_dref()                             # P7 apply @ full res -> dref
        self.gf_b_tex.use(0); self.work.color_attachments[0].use(1)
        self.dref_fbo.use()
        ctx.viewport = (0, 0, self.dref_tex.size[0], self.dref_tex.size[1])
        self.gf_apply_vao.render(moderngl.TRIANGLES)

    def _float_tex(self, size, channels):
        tex = self.ctx.texture(size, channels, dtype="f4")
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        tex.repeat_x = tex.repeat_y = False
        return tex

    # --- the depth pipeline's results ------------------------------------------
    def attach_depth(self, depth_size, flow_size, flow_uv):
        """Size the depth and flow textures to the depth pipeline's (h, w) sizes,
        and wire the GL<->CUDA interop if it is available: an RGBA8 downscale RT
        CUDA reads as the model input (depth_src_gl), and the textures CUDA
        writes the results into (set_result). Call on the GL thread."""
        self.release_depth()
        self.depth_tex.release()
        h, w = depth_size
        fh, fw = flow_size
        self.depth_tex = self._float_tex((w, h), 1)
        self.depth_tex.write(np.full((h, w), 0.5, dtype="f4").tobytes())
        self.flow_tex = [self._float_tex((fw, fh), 4) for _ in range(2)]
        self.flow_uv = flow_uv
        try:
            from .glcuda import GLTexReader, GLTexWriter
            self.idsrc = self.ctx.framebuffer(
                color_attachments=[self.ctx.texture((self.dw, self.dh), 4)])
            self.idsrc.color_attachments[0].filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._ireader = GLTexReader(self.idsrc.color_attachments[0].glo, self.dw, self.dh)
            self._iwriter = GLTexWriter(self.depth_tex.glo, w, h)
            self._fwriters = [GLTexWriter(t.glo, fw, fh, channels=4) for t in self.flow_tex]
        except Exception as e:
            print(f"[braindance] GL<->CUDA interop unavailable, reading back: {e}", flush=True)
            self._release_interop()

    def _release_interop(self):
        for o in (self._ireader, self._iwriter, *self._fwriters):
            if o is not None:
                o.close()
        self._ireader = self._iwriter = None
        self._fwriters = [None, None]
        if self.idsrc is not None:
            self.idsrc.color_attachments[0].release()
            self.idsrc.release()
            self.idsrc = None

    def release(self):
        """Let go of everything CUDA holds of this renderer's textures."""
        self.release_depth()
        self._release_net()

    def release_depth(self):
        self._release_interop()
        for tex in self.flow_tex:
            if tex is not None:
                tex.release()
        self.flow_tex = [None, None]
        self.has_flow = False

    @property
    def interop(self):
        return self._iwriter is not None

    def depth_src_gl(self):
        """Blit the work RT into the RGBA8 interop source and hand CUDA a GPU
        (H, W, 4) uint8 tensor of it — no readback. FBO order (row 0 = bottom)."""
        self.idsrc.use()
        self.ctx.clear()
        self.work.color_attachments[0].use(0)
        self.blit_vao.render(moderngl.TRIANGLES)
        return self._ireader.read()

    def set_result(self, r):
        """Copy a depth.Pipeline result into the depth and flow textures. Its
        tensors are top row first; the textures' row 0 is the bottom. None (the
        pipeline had nothing new) keeps what they hold."""
        if r is None:
            return
        if self.interop:
            self._iwriter.write(r.depth.flip(0))
            if r.flow is not None:
                for eye in range(2):
                    self._fwriters[eye].write(r.flow[eye].permute(1, 2, 0).flip(0))
        else:
            from .depth import to_numpy
            self.depth_tex.write(to_numpy(r.depth.flip(0)).tobytes())
            if r.flow is not None:
                for eye in range(2):
                    self.flow_tex[eye].write(to_numpy(r.flow[eye].permute(1, 2, 0).flip(0)).tobytes())
        self.has_flow = r.flow is not None

    # --- present -------------------------------------------------------------
    def draw(self, sbs_disp, zoom, panx, pany, pointer=None, warp="mesh",
             edge_fill="mirror", edge_cliff=4.0, edge_halo=6, inpaint=None, sharp_mode="off",
             jbu_color=0.5, jbu_reach=1.5, jbu_depth=0.0,
             gf_eps=0.01, gf_radius=4, climit_px=0.0, screen_depth_px=0.0,
             fill_debug=False, depth_view="off"):
        ctx = self.ctx
        disp_ndc = sbs_disp / (self.eye_w / 2.0)
        depth_mode = disp_ndc > 1e-6
        # The learned warp needs its offsets; until the first ones arrive (and in
        # FLAT) the mesh draws. The edge fill and refining serve the mesh alone;
        # in FLAT nothing stretches, so there is nothing to fill. Refining also
        # runs for the depth-debug views.
        use_mlbw = warp == "mlbw" and depth_mode and self.has_flow
        mesh = warp == "mesh"
        do_fill = edge_fill != "off" and depth_mode and mesh
        do_sharp = (sharp_mode != "off") and (depth_mode or depth_view != "off") and mesh

        self.work.color_attachments[0].use(0)
        # edge-aware depth refine -> dref_tex, so the warp samples crisp depth edges.
        # Two backends behind one slot.
        if do_sharp and sharp_mode == "jbu":
            # color-guided joint-bilateral upsample (reach + color + depth terms).
            self._ensure_dref()
            self.depth_tex.use(1)
            self.dref_fbo.use()
            ctx.viewport = (0, 0, self.dref_tex.size[0], self.dref_tex.size[1])
            # gather stride = DEPTH texel (kernel reaches across the low-res edge);
            # color compared per full-res output pixel -> full-res edges.
            self.sharp["dtexel"].value = (1.0 / self.depth_tex.size[0],
                                          1.0 / self.depth_tex.size[1])
            self.sharp["reach"].value = float(jbu_reach)
            # color 0..1 -> color sigma (smaller = tighter = more edge-snapping)
            self.sharp["sigma_c"].value = max(0.03, 0.30 - 0.27 * float(jbu_color))
            # depth 0..1 -> range sigma (0 disables the term; higher = stricter)
            self.sharp["sigma_d"].value = (10.0 if jbu_depth <= 1e-6
                                           else max(0.02, 0.30 - 0.28 * float(jbu_depth)))
            self.sharp_vao.render(moderngl.TRIANGLES)
            depth_src = self.dref_tex
        elif do_sharp and sharp_mode == "guided":
            self._guided(gf_eps, gf_radius)
            depth_src = self.dref_tex
        else:
            depth_src = self.depth_tex

        # Render into the supersampled RT when SSAA is on, then box-downsample to
        # the screen; otherwise draw straight to screen.
        target = self.ss_fbo if self.ss_fbo is not None else self.screen
        target.use()
        s = self.ss
        ew, eh = int(round(self.eye_w * s)), int(round(self.ph * s))

        if depth_view == "2d":
            # flat debug: depth map (refined if SHARP MODE != off) as grayscale, both
            # eyes identical. No warp.
            ctx.clear(0.0, 0.0, 0.0)
            depth_src.use(0)
            for half in range(2):
                ctx.viewport = (half * ew, 0, ew, eh)
                self.depthview_vao.render(moderngl.TRIANGLES)
        elif use_mlbw:
            ctx.clear(0.0, 0.0, 0.0)
            pan_ndc = (panx / self.eye_w * 2.0, -pany / self.ph * 2.0)
            vshift_ndc = screen_depth_px / (self.eye_w / 2.0)
            m = self.mlbw
            m["uv_scale"].value = self.flow_uv
            m["zoom"].value = zoom
            m["pan"].value = pan_ndc
            m["vshift"].value = vshift_ndc
            m["climit"].value = climit_px / self.eye_w
            m["show_depth"].value = 1.0 if depth_view == "3d" else 0.0
            self.work.color_attachments[0].use(0)
            depth_src.use(1)
            for half in range(2):
                ctx.viewport = (half * ew, 0, ew, eh)
                self.flow_tex[half].use(2)
                m["eye_sign"].value = 1.0 if half == 0 else -1.0
                self.mlbw_vao.render(moderngl.TRIANGLES)
            if pointer is not None and pointer[2] and depth_view == "off":
                self._draw_cursor(pointer, disp_ndc, vshift_ndc, zoom, pan_ndc, ew, eh)
        else:
            # warp path — normal color, OR the 3D depth view (depth as grayscale,
            # which still gets parallax and the edge fill). Each eye's mesh goes to
            # its eye targets with the depth test, and the edge fill finds what to
            # replace; then a blend brings each eye to the target with the mirror
            # fill or the network's picture, or a blit does when the fill is off.
            ctx.clear(0.0, 0.0, 0.0)
            self._ensure_eye_rt(ew, eh)
            prog = self.warp
            prog["show_depth"].value = 1.0 if depth_view == "3d" else 0.0
            prog["zoom"].value = zoom
            pan_ndc = (panx / self.eye_w * 2.0, -pany / self.ph * 2.0)
            prog["pan"].value = pan_ndc
            prog["climit"].value = climit_px / (self.eye_w / 2.0)
            vshift_ndc = screen_depth_px / (self.eye_w / 2.0)
            prog["vshift"].value = vshift_ndc
            prog["disp"].value = disp_ndc
            prog["dtexel"].value = 1.0 / depth_src.size[0]
            prog["src_px"].value = ew * zoom
            # the fill's sizes are in present px; SSAA renders s times as many
            self.edge_fill["texel"].value = (1.0 / ew, 1.0 / eh)
            self.edge_fill["cliff_px"].value = float(edge_cliff) * s
            self.edge_fill["feather"].value = 3.0 * s
            self.edge_fill["halo"].value = int(round(edge_halo * s))
            self.edge_mix["texel"].value = (1.0 / ew, 1.0 / eh)
            for half in range(2):
                col, msk = self.eye_fbo[half].color_attachments
                self.eye_fbo[half].use()
                ctx.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
                ctx.enable(moderngl.DEPTH_TEST)
                self.work.color_attachments[0].use(0)
                depth_src.use(1)
                prog["eye_sign"].value = 1.0 if half == 0 else -1.0
                self.warp_vao.render(moderngl.TRIANGLES)
                ctx.disable(moderngl.DEPTH_TEST)
                if do_fill:
                    self.fill_fbo[half].use()
                    msk.use(0)
                    col.use(1)
                    self.edge_fill_vao.render(moderngl.TRIANGLES)
            # the network paints both eyes at once, so it runs between the passes
            source = 0.0
            if fill_debug:
                source = 2.0
            elif do_fill and edge_fill == "network" and inpaint is not None:
                source = 1.0 if self._paint(inpaint, ew, eh) else 0.0
            self.edge_mix["source"].value = source
            if source == 1.0:
                self.net_tex.use(2)
            target.use()
            for half in range(2):
                ctx.viewport = (half * ew, 0, ew, eh)
                self.eye_fbo[half].color_attachments[0].use(0)
                if do_fill:
                    self.fill_fbo[half].color_attachments[0].use(1)
                    self.edge_mix["net_x"].value = (0.5, -0.5) if half == 0 else (0.5, 0.5)
                    self.edge_mix_vao.render(moderngl.TRIANGLES)
                else:
                    self.blit_vao.render(moderngl.TRIANGLES)

            if pointer is not None and pointer[2] and depth_view == "off":
                self._draw_cursor(pointer, disp_ndc, vshift_ndc, zoom, pan_ndc, ew, eh)
        if self.ss_fbo is not None:
            self.screen.use()
            ctx.viewport = (0, 0, self.pw, self.ph)
            self.ss_tex.use(0)
            self.resolve_vao.render(moderngl.TRIANGLES)

    def _draw_cursor(self, pointer, disp_ndc, vshift_ndc, zoom, pan_ndc, ew, eh):
        ctx = self.ctx
        u, v = pointer[0], pointer[1]
        bx = (u * 2 - 1) * zoom + pan_ndc[0]
        by = ((1.0 - v) * 2 - 1) * zoom + pan_ndc[1]   # v=0 top -> +1 top
        # gentle fixed float just in front of the screen plane (~4px/eye), not tied to
        # the scene disparity; rides the SCREEN DEPTH slide so it stays on the plane.
        cdisp = 0.004 if disp_ndc > 0 else 0.0
        asp = self.aspect
        for half in range(2):
            ctx.viewport = (half * ew, 0, ew, eh)
            cx = bx + (1.0 if half == 0 else -1.0) * (cdisp + vshift_ndc)
            self.cursor["center"].value = (cx, by)
            # black outline (slightly larger), then green fill
            self.cursor["col"].value = (0.0, 0.0, 0.0)
            self.cursor["halfsz"].value = (0.013, 0.013 * asp)
            self.cursor_vao.render(moderngl.TRIANGLES)
            self.cursor["col"].value = (0.42, 0.0, 0.55)   # dark purple
            self.cursor["halfsz"].value = (0.010, 0.010 * asp)
            self.cursor_vao.render(moderngl.TRIANGLES)

    def draw_panel(self, xy, t=0.0, appear=1.0, screen_depth_px=0.0, pointer=None):
        """The settings panel in each eye with its top-left at `xy` (eye pixels),
        floated a fixed amount in front of the scene: it rides the SCREEN DEPTH
        plane slide, so it stays in front wherever the scene currently sits.
        `pointer` (x, y in eye pixels) draws the cursor at the panel's depth,
        where it is clicking."""
        disp = PANEL_DISP + screen_depth_px / (self.eye_w / 2.0)
        rect = panel.ndc((self.eye_w, self.ph), xy)
        asp = self.aspect
        for half in range(2):
            sign = 1.0 if half == 0 else -1.0
            self.ctx.viewport = (half * self.eye_w, 0, self.eye_w, self.ph)
            self.panel.draw(rect, t, appear, sign * disp)
            if pointer is None:
                continue
            center = (pointer[0] / self.eye_w * 2 - 1 + sign * disp, 1 - pointer[1] / self.ph * 2)
            self.cursor["center"].value = center
            self.cursor["col"].value = (0.0, 0.0, 0.0)
            self.cursor["halfsz"].value = (0.013, 0.013 * asp)
            self.cursor_vao.render(moderngl.TRIANGLES)
            self.cursor["col"].value = (1.0, 0.16, 0.43)
            self.cursor["halfsz"].value = (0.010, 0.010 * asp)
            self.cursor_vao.render(moderngl.TRIANGLES)
