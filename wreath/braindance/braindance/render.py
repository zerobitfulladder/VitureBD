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
from PIL import Image, ImageDraw, ImageFont

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

# Per-eye present (mesh warp). disp=0 -> FLAT. The VS passes per-vertex depth to
# a geometry shader that can TEAR the mesh (drop triangles spanning a big depth
# jump) so foreground silhouettes pop crisply instead of ramping into the bg.
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
out vec2 v_uv_g;
out float v_depth_g;
void main() {
    float d = texture(depth_tex, in_uv).r;
    vec2 p = in_uv * 2.0 - 1.0;          // uv.y=1 -> +1 (top)
    p *= zoom;
    p += pan;
    float dval = disp * (d - 0.5);       // linear: zero-parallax at mid-depth
    if (climit > 0.0) dval = clamp(dval, -climit, climit);   // comfort clamp (3D pop only)
    p.x += eye_sign * (dval + vshift);   // + SCREEN DEPTH plane slide
    gl_Position = vec4(p, 0.0, 1.0);
    v_uv_g = in_uv;
    v_depth_g = d;
}
"""
# Drop a triangle whose vertices straddle a depth discontinuity larger than
# `tear` (in normalized depth). tear huge (1e9) => never tears (continuous warp).
WARP_GS = """
#version 330
layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;
in vec2 v_uv_g[];
in float v_depth_g[];
uniform float tear;      // spread above which a triangle is a depth cliff
uniform int   fill;      // 0 = tear pass (drop cliffs); 1 = background-stretch fill
out vec2 v_uv;
out float v_d;
void main() {
    float a = v_depth_g[0], b = v_depth_g[1], c = v_depth_g[2];
    float spread = max(max(abs(a - b), abs(b - c)), abs(a - c));
    bool cliff = spread > tear;
    if (cliff && fill == 0) return;      // tear pass: tear the cliff triangle away
    // background-stretch fill: paint a cliff triangle with its FARTHEST (background)
    // vertex's color, so the disocclusion gap reads as stretched background instead
    // of a foreground->background ramp. (larger depth = nearer, so background = min.)
    vec2 bg_uv = v_uv_g[0]; float bgd = a;
    if (b < bgd) { bgd = b; bg_uv = v_uv_g[1]; }
    if (c < bgd) { bgd = c; bg_uv = v_uv_g[2]; }
    for (int i = 0; i < 3; i++) {
        gl_Position = gl_in[i].gl_Position;
        v_uv = (cliff && fill == 1) ? bg_uv : v_uv_g[i];
        v_d  = v_depth_g[i];
        EmitVertex();
    }
    EndPrimitive();
}
"""
WARP_FS = """
#version 330
in vec2 v_uv;
in float v_d;
uniform sampler2D color_tex;
uniform float show_depth;     // 1 -> draw the depth map (grayscale) instead of color
out vec4 f;
void main() {
    f = (show_depth > 0.5) ? vec4(vec3(v_d), 1.0) : texture(color_tex, v_uv);
}
"""

# Same warp WITHOUT the geometry shader, for the no-tear path (the common case).
# The GS is only needed for tearing and is the dominant present cost at high grid;
# with tear off it's a pure passthrough. Derived from WARP_VS so the displacement
# math stays single-sourced — only the outputs are renamed to feed WARP_FS directly
# (v_uv/v_d) instead of the GS-bound *_g names.
WARP_VS_NOGS = (WARP_VS
                .replace("out vec2 v_uv_g;", "out vec2 v_uv;")
                .replace("out float v_depth_g;", "out float v_d;")
                .replace("v_uv_g = in_uv;", "v_uv = in_uv;")
                .replace("v_depth_g = d;", "v_d = d;"))

HUD_VS = """
#version 330
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() { gl_Position = vec4(in_pos, 0.0, 1.0); v_uv = in_uv; }
"""
HUD_FS = """
#version 330
in vec2 v_uv;
uniform sampler2D hud_tex;
out vec4 f;
void main() { f = texture(hud_tex, v_uv); }
"""

# Settings OSD panel: a textured quad floated in front via a per-eye x offset.
PANEL_VS = """
#version 330
in vec2 in_pos;
in vec2 in_uv;
uniform vec2 off;      // per-eye NDC offset (disparity)
out vec2 v_uv;
void main() { gl_Position = vec4(in_pos + off, 0.0, 1.0); v_uv = in_uv; }
"""

# Animated OSD: samples the cached panel bitmap with cyberpunk effects — chromatic
# aberration (subtle, amplified during a glitch burst), occasional horizontal-slice
# glitch + noise flash, a breathing magenta edge-glow, and a noise "materialize"
# dissolve on open (`appear` 0->1). No scanlines, no flicker. `time` = small seconds.
PANEL_FS = """
#version 330
in vec2 v_uv;
uniform sampler2D hud_tex;
uniform float time;
uniform float appear;     // 0..1 open materialize
out vec4 f;
float h1(float x){ return fract(sin(x * 12.9898) * 43758.5453123); }
float h2(vec2 p){ return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453123); }
void main() {
    vec2 uv = v_uv;
    // occasional brief glitch burst
    float win = floor(time * 3.0);
    float burst = step(0.90, h1(win)) * (0.4 + 0.6 * h1(win + 7.0));
    float slice = floor(uv.y * 26.0);
    float jit = step(0.55, h1(slice * 1.7 + floor(time * 22.0)));
    uv.x += burst * (h1(slice + floor(time * 22.0)) - 0.5) * 0.08 * jit;
    // chromatic aberration (subtle base + burst)
    float ca = 0.0016 + burst * 0.006;
    float r = texture(hud_tex, clamp(uv + vec2(ca, 0.0), 0.0, 1.0)).r;
    vec4  m = texture(hud_tex, clamp(uv, 0.0, 1.0));
    float b = texture(hud_tex, clamp(uv - vec2(ca, 0.0), 0.0, 1.0)).b;
    vec4 col = vec4(r, m.g, b, m.a);
    // noise flash during a burst
    col.rgb += burst * (h2(uv * vec2(800.0, 460.0) + time) - 0.5) * 0.18 * col.a;
    // breathing magenta edge-glow on the panel border
    vec2 dd = min(v_uv, 1.0 - v_uv);
    float edge = 1.0 - smoothstep(0.0, 0.05, min(dd.x, dd.y));
    float pulse = 0.55 + 0.45 * sin(time * 3.0);
    col.rgb += edge * pulse * vec3(1.0, 0.16, 0.43) * 0.6 * col.a;
    // open: glitchy scan-lock reveal. Horizontal bands snap in OUT OF ORDER with a
    // sideways tear + RGB split + static, each resolving into the clean panel — reads
    // as a signal locking in line by line, not a blocky square slide.
    if (appear < 0.999) {
        float row  = floor(v_uv.y * 30.0);
        float thr  = h1(row * 1.93 + 0.5);                  // per-row reveal threshold
        float lock = smoothstep(thr, thr + 0.18, appear);   // 0 scrambled .. 1 locked
        float unl  = 1.0 - lock;
        float j    = (h1(row + floor(time * 28.0)) - 0.5) * unl;       // sideways tear
        vec2  tuv  = clamp(v_uv + vec2(j * 0.6, 0.0), 0.0, 1.0);
        float sp   = 0.03 * unl;                                       // RGB split width
        vec4  ts   = texture(hud_tex, tuv);
        vec3  tc   = vec3(texture(hud_tex, clamp(tuv + vec2(sp, 0.0), 0.0, 1.0)).r,
                          ts.g,
                          texture(hud_tex, clamp(tuv - vec2(sp, 0.0), 0.0, 1.0)).b);
        tc += (h2(v_uv * vec2(700.0, 40.0) + time) - 0.5) * 0.6 * unl;  // static hash
        col.rgb = mix(tc, col.rgb, lock);
        col.a   = mix(ts.a, col.a, lock);
        float front = smoothstep(0.05, 0.0, abs(appear - thr)) * unl;   // tracking spark
        col.rgb += front * vec3(0.1, 0.95, 1.0) * 0.8;
        col.a *= smoothstep(0.0, 0.12, appear);             // global fade floor
    }
    f = col;
}
"""

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

# Disocclusion inpaint (texture-continuation). The per-eye warp leaves torn gaps as
# holes (alpha 0). Stereo disparity is horizontal, so a gap is a 1-D strip with
# BACKGROUND on a known side (left eye pulls left, right eye pulls right => `dir`).
# For each hole pixel we walk `dir` to the background edge, then CONTINUE the
# background by reflecting it across the edge into the gap (edge maps to itself =>
# seamless, and the bg texture/gradient carries on instead of a flat smear).
INPAINT_FS = """
#version 330
in vec2 v_uv;
uniform sampler2D src;     // per-eye warped color; alpha 0 = disocclusion hole
uniform vec2 texel;        // 1 / (eye_w, eye_h) at present res
uniform float dir;         // background march direction in x: -1 (left eye) / +1
out vec4 f;
const int MAXW = 256;      // widest hole we attempt to fill (texels)
void main() {
    vec4 c = texture(src, v_uv);
    if (c.a > 0.5) { f = vec4(c.rgb, 1.0); return; }     // not a hole -> passthrough
    vec2 stp = vec2(dir * texel.x, 0.0);
    int k = 0;                                           // steps to background edge
    for (int i = 1; i <= MAXW; i++) {
        if (texture(src, v_uv + float(i) * stp).a > 0.5) { k = i; break; }
    }
    if (k == 0) { f = vec4(c.rgb, 0.0); return; }         // unfilled hole -> a=0
    // reflect across the edge: hole pixel k texels inside <- bg pixel k texels past
    vec4 b = texture(src, v_uv + float(2 * k) * stp);
    if (b.a <= 0.5) b = texture(src, v_uv + float(k) * stp);   // thin bg -> edge px
    f = vec4(b.rgb, 0.0);                                 // filled -> mark a=0
}
"""

# Smooth the inpainted region to break up the horizontal streaks that 1-D
# continuation leaves. Vertical-biased box over ONLY the filled pixels (alpha 0),
# so the high vertical-frequency stripes average out while foreground and crisp
# background edges (alpha 1) are never pulled in.
INPAINT_SMOOTH_FS = """
#version 330
in vec2 v_uv;
uniform sampler2D src;     // fill RT: rgb=color, a=1 original / a=0 inpainted
uniform vec2 texel;
out vec4 f;
const int RY = 6;          // vertical radius (kills horizontal streaks)
const int RX = 1;          // small horizontal radius (keep bg continuation)
void main() {
    vec4 c = texture(src, v_uv);
    if (c.a > 0.5) { f = vec4(c.rgb, 1.0); return; }     // original -> passthrough
    vec3 sum = vec3(0.0); float w = 0.0;
    for (int dy = -RY; dy <= RY; dy++)
        for (int dx = -RX; dx <= RX; dx++) {
            vec4 s = texture(src, v_uv + vec2(float(dx), float(dy)) * texel);
            if (s.a < 0.5) { sum += s.rgb; w += 1.0; }   // blend filled pixels only
        }
    f = vec4(w > 0.0 ? sum / w : c.rgb, 1.0);
}
"""

# Edge-aware depth UPSAMPLE: a COLOR-guided joint-bilateral filter run at HIGH
# output resolution. For each output texel, average low-res depth samples weighted
# by spatial distance AND color similarity to the center (full-res color guide) —
# so the soft, low-res DA2 edge snaps onto the real silhouette at high res (kills
# the laddery edges, and gives the mesh a real cliff to tear). dtexel = OUTPUT
# texel (offsets walk the full-res guide); sigma_c small = tighter edge-snapping.
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
        self.pw, self.ph, self.eye_w = pw, ph, eye_w
        self.rw, self.rh = region
        self.eye_h = ph
        self.ss = max(1.0, float(ss))   # present supersampling factor (1.0 = off)

        # work-area RT (composited desktop), v=1 = top
        self.work = ctx.framebuffer(color_attachments=[ctx.texture((self.rw, self.rh), 3)])
        self.work.color_attachments[0].filter = (moderngl.LINEAR, moderngl.LINEAR)

        # downscaled RT for depth-model readback
        dw = max(14, int(round(depth_h * self.rw / self.rh / 14)) * 14)
        self.dw, self.dh = dw, depth_h
        self.dsrc = ctx.framebuffer(color_attachments=[ctx.texture((dw, depth_h), 3)])
        self.dsrc.color_attachments[0].filter = (moderngl.LINEAR, moderngl.LINEAR)
        # double-buffered PBOs for async readback (no glReadPixels stall)
        self._pbos = [ctx.buffer(reserve=dw * depth_h * 3) for _ in range(2)]
        self._pbo_i = 0
        self._pbo_primed = False

        # depth texture (uploaded each time a new map arrives)
        self.depth_tex = ctx.texture((dw, depth_h), 1, dtype="f4")
        self.depth_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.depth_tex.write(np.full((depth_h, dw), 0.5, dtype="f4").tobytes())

        # GL<->CUDA interop (lazy; the sync depth path enables it). None when off.
        self.idsrc = None
        self._ireader = self._iwriter = None

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

        gy = max(1, grid * self.rh // self.rw)
        uv, idx = _grid(grid, gy)
        uvbuf, idxbuf = ctx.buffer(uv.tobytes()), ctx.buffer(idx.tobytes())
        self.warp = ctx.program(vertex_shader=WARP_VS, geometry_shader=WARP_GS,
                                fragment_shader=WARP_FS)
        self.warp["color_tex"] = 0
        self.warp["depth_tex"] = 1
        self.warp["tear"] = 1e9          # default: no tearing (continuous)
        self.warp["fill"] = 0            # 1 -> background-stretch fill (tear pass 1)
        self.warp["show_depth"] = 0.0    # 1 -> render depth map as grayscale (3D view)
        self.warp["climit"] = 0.0        # comfort disparity clamp (NDC); 0 = off
        self.warp["vshift"] = 0.0        # SCREEN DEPTH: constant per-eye plane slide (NDC)
        self.warp_vao = ctx.vertex_array(
            self.warp, [(uvbuf, "2f", "in_uv")], idxbuf)
        # GS-free twin for the no-tear path (the common case): the geometry shader is
        # only needed for tearing and is the dominant present cost at high grid.
        # Shares the mesh buffers; pixel-identical to the GS warp when tear is off.
        self.warp_nogs = ctx.program(vertex_shader=WARP_VS_NOGS, fragment_shader=WARP_FS)
        self.warp_nogs["color_tex"] = 0
        self.warp_nogs["depth_tex"] = 1
        self.warp_nogs["show_depth"] = 0.0
        self.warp_nogs["climit"] = 0.0
        self.warp_nogs["vshift"] = 0.0
        self.warp_nogs_vao = ctx.vertex_array(
            self.warp_nogs, [(uvbuf, "2f", "in_uv")], idxbuf)

        # HUD
        self.HUD_W, self.HUD_H = 680, 40
        try:
            self.font = ImageFont.truetype("DejaVuSans.ttf", 22)
        except Exception:
            self.font = ImageFont.load_default()
        self.hud = ctx.program(vertex_shader=HUD_VS, fragment_shader=HUD_FS)
        self.hud["hud_tex"] = 0
        hwn, hhn = (self.HUD_W / eye_w) * 2.0, (self.HUD_H / ph) * 2.0
        x0, y1 = -0.97, 0.97; x1, y0 = x0 + hwn, y1 - hhn
        quad = np.array([x0, y1, 0, 0, x1, y1, 1, 0, x0, y0, 0, 1,
                         x1, y1, 1, 0, x1, y0, 1, 1, x0, y0, 0, 1], dtype="f4")
        self.hud_vao = ctx.vertex_array(self.hud, [(ctx.buffer(quad.tobytes()), "2f 2f", "in_pos", "in_uv")])
        self.hud_tex = ctx.texture((self.HUD_W, self.HUD_H), 4)
        self.hud_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.set_hud("starting...")

        # custom cursor: a small classic arrow pointing up-left (tip = hotspot at
        # local origin; left edge straight down, diagonal back up-right), constant
        # on-screen size, with its own disparity to float above.
        ctri = np.array([0.0, 0.0,  0.0, -1.4,  0.95, -0.95], dtype="f4")
        self.cursor = ctx.program(vertex_shader=CURSOR_VS, fragment_shader=CURSOR_FS)
        self.cursor_vao = ctx.vertex_array(self.cursor, [(ctx.buffer(ctri.tobytes()), "2f", "in_p")])
        self.aspect = eye_w / ph

        # --- settings OSD panel (toggled by Super+Alt+S) ---------------------
        # compact, anchored BOTTOM-RIGHT so the scene centre stays visible while
        # the panel is open (test 3D live); tabbed single column (see set_panel).
        self.PANEL_W, self.PANEL_H = 640, 436
        self.panel_disp = 0.018            # floats in front of the scene
        try:
            self.pfont = ImageFont.truetype("DejaVuSansMono.ttf", 21)
            self.pfont_b = ImageFont.truetype("DejaVuSansMono-Bold.ttf", 26)
            self.pfont_s = ImageFont.truetype("DejaVuSansMono.ttf", 16)
        except Exception:
            self.pfont = self.pfont_b = self.pfont_s = ImageFont.load_default()
        self.panel = ctx.program(vertex_shader=PANEL_VS, fragment_shader=PANEL_FS)
        self.panel["hud_tex"] = 0
        pwn, phn = (self.PANEL_W / eye_w), (self.PANEL_H / ph)   # half-extents in NDC
        mx, my = 0.04, 0.05                # right / bottom margins (NDC)
        cx, cy = 1.0 - mx - pwn, -1.0 + my + phn
        x0, x1, y0, y1 = cx - pwn, cx + pwn, cy - phn, cy + phn
        pquad = np.array([x0, y1, 0, 0, x1, y1, 1, 0, x0, y0, 0, 1,
                          x1, y1, 1, 0, x1, y0, 1, 1, x0, y0, 0, 1], dtype="f4")
        self.panel_vao = ctx.vertex_array(
            self.panel, [(ctx.buffer(pquad.tobytes()), "2f 2f", "in_pos", "in_uv")])
        self.panel_tex = ctx.texture((self.PANEL_W, self.PANEL_H), 4)
        self.panel_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.panel_tex.repeat_x = False     # clamp: CA/glitch sample past the edge
        self.panel_tex.repeat_y = False
        self._panel_sig = None

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

        # disocclusion inpaint: per-eye warp RT (color+hole mask) + fill pass
        self.inpaint = ctx.program(vertex_shader=BLIT_VS, fragment_shader=INPAINT_FS)
        self.inpaint["src"] = 0
        self.inpaint_vao = ctx.vertex_array(
            self.inpaint, [(ctx.buffer(unit.tobytes()), "2f", "in_uv")])
        self.smooth = ctx.program(vertex_shader=BLIT_VS, fragment_shader=INPAINT_SMOOTH_FS)
        self.smooth["src"] = 0
        self.smooth_vao = ctx.vertex_array(
            self.smooth, [(ctx.buffer(unit.tobytes()), "2f", "in_uv")])
        self.warp_tex = None
        self.warp_fbo = None
        self.fill_tex = None
        self.fill_fbo = None

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

    def _ensure_warp_rt(self, ew, eh):
        """Per-eye RGBA RT the warp renders into for inpaint mode: RGB = warped
        color, A = coverage (0 at torn disocclusion holes). NEAREST + clamp so the
        hole mask stays binary and the horizontal march can't wrap across eyes."""
        if self.warp_tex is not None and self.warp_tex.size == (ew, eh):
            return
        if self.warp_fbo is not None:
            self.warp_fbo.release()
            self.warp_tex.release()
            self.fill_fbo.release()
            self.fill_tex.release()
        self.warp_tex = self.ctx.texture((ew, eh), 4)
        self.fill_tex = self.ctx.texture((ew, eh), 4)   # inpaint output (a=0 = filled)
        for t in (self.warp_tex, self.fill_tex):
            t.filter = (moderngl.NEAREST, moderngl.NEAREST)
            t.repeat_x = False
            t.repeat_y = False
        self.warp_fbo = self.ctx.framebuffer(color_attachments=[self.warp_tex])
        self.fill_fbo = self.ctx.framebuffer(color_attachments=[self.fill_tex])

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

    def set_depth(self, depth_np):
        # depth_np is top-down (from a top-down frame); store v=1=top to match color.
        # Adapt the texture to the backend's output size (mesh samples by UV).
        h, w = depth_np.shape
        if (w, h) != self.depth_tex.size:
            self.depth_tex.release()
            self.depth_tex = self.ctx.texture((w, h), 1, dtype="f4")
            self.depth_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        d = np.ascontiguousarray(depth_np[::-1].astype("f4"))
        self.depth_tex.write(d.tobytes())

    # --- GL<->CUDA interop (zero-copy depth path) ---------------------------
    def setup_interop(self, out_h, out_w):
        """Wire the GL<->CUDA depth path: an RGBA8 downscale RT that CUDA reads as
        the model input, and a depth texture (sized to the model output) that CUDA
        writes into. Call on the GL thread after the backend is built; re-call if
        the backend's output size changes. Raises if interop is unavailable."""
        from .glcuda import GLTexReader, GLTexWriter
        self.teardown_interop()
        self.idsrc = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((self.dw, self.dh), 4)])
        self.idsrc.color_attachments[0].filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._ireader = GLTexReader(self.idsrc.color_attachments[0].glo, self.dw, self.dh)
        # depth texture fixed at the model output size; CUDA writes straight into it
        self.depth_tex.release()
        self.depth_tex = self.ctx.texture((out_w, out_h), 1, dtype="f4")
        self.depth_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.depth_tex.write(np.full((out_h, out_w), 0.5, dtype="f4").tobytes())
        self._iwriter = GLTexWriter(self.depth_tex.glo, out_w, out_h)

    def teardown_interop(self):
        for a in ("_ireader", "_iwriter"):
            o = getattr(self, a, None)
            if o is not None:
                o.close()
                setattr(self, a, None)
        if self.idsrc is not None:
            self.idsrc.color_attachments[0].release()
            self.idsrc.release()
            self.idsrc = None

    def depth_src_interop(self):
        """Blit the work RT into the RGBA8 interop source and hand CUDA a GPU
        (H, W, 4) uint8 tensor of it — no readback. FBO order (row 0 = bottom)."""
        self.idsrc.use()
        self.ctx.clear()
        self.work.color_attachments[0].use(0)
        self.blit_vao.render(moderngl.TRIANGLES)
        return self._ireader.read()

    def write_depth_interop(self, depth_hw):
        """Copy a GPU (H, W) f32 depth straight into the GL depth texture."""
        self._iwriter.write(depth_hw)

    # --- present -------------------------------------------------------------
    def draw(self, sbs_disp, zoom, panx, pany, pointer=None,
             tear=False, tear_thresh=0.08, tear_fill="inpaint", sharp_mode="off",
             jbu_color=0.5, jbu_reach=1.5, jbu_depth=0.0,
             gf_eps=0.01, gf_radius=4, climit_px=0.0, screen_depth_px=0.0,
             tear_debug=False, depth_view="off"):
        ctx = self.ctx
        disp_ndc = sbs_disp / (self.eye_w / 2.0)
        depth_mode = disp_ndc > 1e-6
        # Tearing only makes sense with disparity (in FLAT it would punch holes in a
        # flat image); sharpening also runs for the depth-debug views.
        do_tear = tear and depth_mode
        do_sharp = (sharp_mode != "off") and (depth_mode or depth_view != "off")

        self.work.color_attachments[0].use(0)
        # edge-aware depth refine -> dref_tex, so the warp samples crisp depth edges
        # (which is what lets it tear cleanly). Two backends behind one slot.
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
        target = self.ss_fbo if self.ss_fbo is not None else ctx.screen
        target.use()
        s = self.ss
        ew, eh = int(round(self.eye_w * s)), int(round(self.ph * s))

        if depth_view == "2d":
            # flat debug: depth map (refined if SHARP MODE != off) as grayscale, both
            # eyes identical. No warp, no tearing.
            ctx.clear(0.0, 0.0, 0.0)
            depth_src.use(0)
            for half in range(2):
                ctx.viewport = (half * ew, 0, ew, eh)
                self.depthview_vao.render(moderngl.TRIANGLES)
        else:
            # warp path — normal color, OR the 3D depth view (depth as grayscale,
            # which still gets parallax + tearing + tear-debug). The geometry-shader
            # warp is only needed for tearing; the no-tear path (common case) uses the
            # GS-free twin to skip the per-triangle GS cost.
            do_gs = do_tear
            prog = self.warp if do_gs else self.warp_nogs
            vao = self.warp_vao if do_gs else self.warp_nogs_vao
            prog["show_depth"].value = 1.0 if depth_view == "3d" else 0.0
            # tear-debug: torn gaps show bright magenta instead of the background fill
            if do_tear and tear_debug:
                ctx.clear(0.55, 0.0, 0.35)
            else:
                ctx.clear(0.0, 0.0, 0.0)
            self.work.color_attachments[0].use(0)
            depth_src.use(1)
            prog["zoom"].value = zoom
            pan_ndc = (panx / self.eye_w * 2.0, -pany / self.ph * 2.0)
            prog["pan"].value = pan_ndc
            prog["climit"].value = climit_px / (self.eye_w / 2.0)
            vshift_ndc = screen_depth_px / (self.eye_w / 2.0)
            prog["vshift"].value = vshift_ndc
            # inpaint mode renders each eye's torn warp (real holes) into a per-eye
            # RT, then fills the holes; other modes warp straight into the target.
            inpaint_mode = do_tear and tear_fill == "inpaint" and not tear_debug
            if inpaint_mode:
                self._ensure_warp_rt(ew, eh)
            for half in range(2):
                sign = 1.0 if half == 0 else -1.0
                if inpaint_mode:
                    # 1) warp this eye with tearing -> color + hole mask (alpha 0)
                    self.warp_fbo.use()
                    ctx.viewport = (0, 0, ew, eh)
                    ctx.clear(0.0, 0.0, 0.0, 0.0)
                    self.work.color_attachments[0].use(0)
                    depth_src.use(1)
                    prog["eye_sign"].value = sign
                    prog["disp"].value = disp_ndc
                    prog["fill"].value = 0
                    prog["tear"].value = float(tear_thresh)
                    vao.render(moderngl.TRIANGLES)
                    # 2) inpaint holes -> fill RT (marks filled pixels a=0). Background
                    # lies left for the left eye (eye_sign +1), right for the right eye.
                    self.fill_fbo.use()
                    ctx.viewport = (0, 0, ew, eh)
                    self.warp_tex.use(0)
                    self.inpaint["texel"].value = (1.0 / ew, 1.0 / eh)
                    self.inpaint["dir"].value = -1.0 if half == 0 else 1.0
                    self.inpaint_vao.render(moderngl.TRIANGLES)
                    # 3) smooth just the filled pixels -> target eye viewport (breaks
                    # up the horizontal streaks from 1-D continuation).
                    target.use()
                    ctx.viewport = (half * ew, 0, ew, eh)
                    self.fill_tex.use(0)
                    self.smooth["texel"].value = (1.0 / ew, 1.0 / eh)
                    self.smooth_vao.render(moderngl.TRIANGLES)
                    continue
                ctx.viewport = (half * ew, 0, ew, eh)
                prog["eye_sign"].value = sign
                prog["disp"].value = disp_ndc
                if do_tear:
                    if not tear_debug:                   # pass 1: fill behind the tear
                        if tear_fill == "stretch":       # background-stretch the gap
                            prog["fill"].value = 1
                            prog["tear"].value = float(tear_thresh)
                        else:                            # "ramp": legacy continuous
                            prog["fill"].value = 0
                            prog["tear"].value = 1e9
                        vao.render(moderngl.TRIANGLES)
                    prog["fill"].value = 0               # pass 2: crisp torn foreground
                    prog["tear"].value = float(tear_thresh)
                    vao.render(moderngl.TRIANGLES)
                else:
                    vao.render(moderngl.TRIANGLES)       # GS-free warp: no fill/tear

            if pointer is not None and pointer[2] and depth_view == "off":
                self._draw_cursor(pointer, disp_ndc, vshift_ndc, zoom, pan_ndc, ew, eh)
        if self.ss_fbo is not None:
            ctx.screen.use()
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

    def draw_hud(self):
        ctx = self.ctx
        ctx.enable(moderngl.BLEND)
        self.hud_tex.use(0)
        for half in range(2):
            ctx.viewport = (half * self.eye_w, 0, self.eye_w, self.ph)
            self.hud_vao.render(moderngl.TRIANGLES)
        ctx.disable(moderngl.BLEND)

    def set_hud(self, text):
        img = Image.new("RGBA", (self.HUD_W, self.HUD_H), (0, 0, 0, 140))
        ImageDraw.Draw(img).text((8, 8), text, font=self.font, fill=(120, 255, 140, 255))
        self.hud_tex.write(np.asarray(img).tobytes())

    # --- settings OSD panel --------------------------------------------------
    # Night City palette (mirrors ~/Scripts/palette_cyberpunk.py + picker.py).
    _PAL = {
        "bg": (10, 4, 24, 232), "border": (255, 42, 109), "magenta": (255, 42, 109),
        "cyan": (5, 217, 232), "yellow": (252, 238, 12), "lime": (204, 255, 0),
        "violet": (185, 103, 255), "fg": (160, 168, 200), "fg_hi": (200, 208, 232),
        "muted": (90, 74, 120), "error": (255, 0, 60),
    }

    def _val_color(self, row):
        P = self._PAL
        if not row["editable"]:
            return P["muted"]
        if row["kind"] == "bool":
            return P["lime"] if row["value"] == "ON" else P["muted"]
        if row["kind"] == "enum":
            return P["cyan"]
        return P["yellow"]

    def set_panel(self, tab_names, tab_idx, rows, fps):
        """Rebuild the OSD texture (only when something changes). Compact, tabbed,
        single column — anchored bottom-right so it doesn't occlude the scene.
          rows: current tab only — section {section:True,label} or
                setting {section:False,label,value,kind,editable,selected}.
        """
        sig = (tab_idx, tuple(tab_names),
               tuple((r.get("section", False), r.get("label"), r.get("value"),
                      r.get("selected")) for r in rows), int(fps))
        if sig == self._panel_sig:
            return
        self._panel_sig = sig
        P = self._PAL
        W, H = self.PANEL_W, self.PANEL_H
        img = Image.new("RGBA", (W, H), P["bg"])
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, W - 1, H - 1], outline=P["border"], width=2)

        pad = 28
        # title
        d.text((pad, 12), "BRAINDANCE", font=self.pfont_b, fill=P["magenta"])
        tw = d.textlength("BRAINDANCE", font=self.pfont_b)
        d.text((pad + tw + 14, 18), "v4.0-NC", font=self.pfont_s, fill=P["yellow"])

        # tab strip
        tx, ty = pad, 46
        for i, name in enumerate(tab_names):
            w = d.textlength(name, font=self.pfont) + 22
            if i == tab_idx:
                d.rectangle([tx, ty, tx + w, ty + 30], fill=(36, 24, 64, 255))
                d.rectangle([tx, ty + 28, tx + w, ty + 30], fill=P["magenta"])
                d.text((tx + 11, ty + 4), name, font=self.pfont, fill=P["fg_hi"])
            else:
                d.text((tx + 11, ty + 4), name, font=self.pfont, fill=P["muted"])
            tx += w + 5
        d.line([pad, ty + 36, W - pad, ty + 36], fill=P["muted"], width=1)

        # settings column (single, full panel width)
        cw, lh = W - 2 * pad, 34
        y = ty + 48
        for r in rows:
            if r.get("section"):
                y += 6
                d.rectangle([pad, y, pad + cw, y + 24], fill=(26, 16, 48, 255))
                d.rectangle([pad, y, pad + 5, y + 24], fill=P["cyan"])
                d.text((pad + 14, y + 1), r["label"], font=self.pfont_s, fill=P["cyan"])
                y += 24 + 6
                continue
            sel = r["selected"]
            if sel:
                d.rectangle([pad, y - 4, pad + cw, y + 26], fill=(36, 24, 64, 255))
                d.rectangle([pad, y - 4, pad + 5, y + 26], fill=P["magenta"])
            d.text((pad + 16, y), r["label"], font=self.pfont,
                   fill=P["fg_hi"] if sel else P["fg"])
            val = r["value"]
            col = self._val_color(r)
            vx = pad + cw - 150
            if sel and r["editable"]:
                d.text((vx - 22, y), "‹", font=self.pfont, fill=P["magenta"])
                d.text((vx, y), val, font=self.pfont, fill=col)
                w = d.textlength(val, font=self.pfont)
                d.text((vx + w + 8, y), "›", font=self.pfont, fill=P["magenta"])
            else:
                d.text((vx, y), val, font=self.pfont, fill=col)
            y += lh

        # footer
        d.line([pad, H - 44, W - pad, H - 44], fill=P["muted"], width=1)
        fcol = P["lime"] if fps >= 55 else P["yellow"] if fps >= 30 else P["error"]
        d.text((pad, H - 34), "FPS", font=self.pfont_s, fill=P["fg"])
        d.text((pad + 46, H - 34), f"{fps:.0f}", font=self.pfont_s, fill=fcol)
        d.text((W - 300, H - 34), "↑↓ ←→   ⇆ tab   ⏎ close",
               font=self.pfont_s, fill=P["muted"])
        self.panel_tex.write(np.asarray(img).tobytes())

    def draw_panel(self, t=0.0, appear=1.0, screen_depth_px=0.0):
        ctx = self.ctx
        ctx.enable(moderngl.BLEND)
        self.panel["time"].value = float(t)
        self.panel["appear"].value = float(appear)
        self.panel_tex.use(0)
        # ride the SCREEN DEPTH plane slide so the panel floats a fixed amount in
        # front of the scene wherever the scene currently sits (not a fixed depth).
        vshift_ndc = screen_depth_px / (self.eye_w / 2.0)
        disp = self.panel_disp + vshift_ndc
        for half in range(2):
            ctx.viewport = (half * self.eye_w, 0, self.eye_w, self.ph)
            self.panel["off"].value = ((1.0 if half == 0 else -1.0) * disp, 0.0)
            self.panel_vao.render(moderngl.TRIANGLES)
        ctx.disable(moderngl.BLEND)
