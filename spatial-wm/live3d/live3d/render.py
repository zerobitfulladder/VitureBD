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

# Per-eye present (mesh warp). disp=0 -> FLAT.
WARP_VS = """
#version 330
in vec2 in_uv;
uniform sampler2D depth_tex;
uniform float disp;      // max disparity (NDC x)
uniform float conv;      // convergence depth 0..1
uniform float eye_sign;  // +1 / -1
uniform float zoom;
uniform vec2  pan;       // NDC
out vec2 v_uv;
void main() {
    float d = texture(depth_tex, in_uv).r;
    vec2 p = in_uv * 2.0 - 1.0;          // uv.y=1 -> +1 (top)
    p *= zoom;
    p += pan;
    p.x += eye_sign * disp * (d - conv);
    gl_Position = vec4(p, 0.0, 1.0);
    v_uv = in_uv;
}
"""
WARP_FS = """
#version 330
in vec2 v_uv;
uniform sampler2D color_tex;
out vec4 f;
void main() { f = texture(color_tex, v_uv); }
"""

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
    def __init__(self, ctx, pw, ph, eye_w, region, grid=256, depth_h=392):
        self.ctx = ctx
        self.pw, self.ph, self.eye_w = pw, ph, eye_w
        self.rw, self.rh = region
        self.eye_h = ph

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

        # programs / geometry
        unit = np.array([0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1], dtype="f4")
        self.comp = ctx.program(vertex_shader=COMPOSITE_VS, fragment_shader=COMPOSITE_FS)
        self.comp["region"].value = (float(self.rw), float(self.rh))
        self.comp_vao = ctx.vertex_array(self.comp, [(ctx.buffer(unit.tobytes()), "2f", "in_uv")])

        self.blit = ctx.program(vertex_shader=BLIT_VS, fragment_shader=BLIT_FS)
        self.blit_vao = ctx.vertex_array(self.blit, [(ctx.buffer(unit.tobytes()), "2f", "in_uv")])

        gy = max(1, grid * self.rh // self.rw)
        uv, idx = _grid(grid, gy)
        self.warp = ctx.program(vertex_shader=WARP_VS, fragment_shader=WARP_FS)
        self.warp["color_tex"] = 0
        self.warp["depth_tex"] = 1
        self.warp_vao = ctx.vertex_array(
            self.warp, [(ctx.buffer(uv.tobytes()), "2f", "in_uv")], ctx.buffer(idx.tobytes()))

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

    # --- present -------------------------------------------------------------
    def draw(self, sbs_disp, conv, zoom, panx, pany, pointer=None):
        ctx = self.ctx
        ctx.screen.use()
        ctx.clear(0.0, 0.0, 0.0)
        self.work.color_attachments[0].use(0)
        self.depth_tex.use(1)
        self.warp["conv"].value = conv
        self.warp["zoom"].value = zoom
        pan_ndc = (panx / self.eye_w * 2.0, -pany / self.ph * 2.0)
        self.warp["pan"].value = pan_ndc
        disp_ndc = sbs_disp / (self.eye_w / 2.0)
        # Per-eye viewport already confines rendering (NDC clips to the viewport
        # rect), so no scissor needed.
        for half in range(2):
            x0 = half * self.eye_w
            ctx.viewport = (x0, 0, self.eye_w, self.ph)
            self.warp["eye_sign"].value = 1.0 if half == 0 else -1.0
            self.warp["disp"].value = disp_ndc
            self.warp_vao.render(moderngl.TRIANGLES)
        if pointer is not None and pointer[2]:
            self._draw_cursor(pointer, disp_ndc, conv, zoom, pan_ndc)

    def _draw_cursor(self, pointer, disp_ndc, conv, zoom, pan_ndc):
        ctx = self.ctx
        u, v = pointer[0], pointer[1]
        bx = (u * 2 - 1) * zoom + pan_ndc[0]
        by = ((1.0 - v) * 2 - 1) * zoom + pan_ndc[1]   # v=0 top -> +1 top
        # gentle fixed float just in front of the screen plane (~4px/eye), not
        # tied to the scene disparity so it never pops out too far
        cdisp = 0.004 if disp_ndc > 0 else 0.0
        asp = self.aspect
        for half in range(2):
            ctx.viewport = (half * self.eye_w, 0, self.eye_w, self.ph)
            cx = bx + (1.0 if half == 0 else -1.0) * cdisp
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
