"""space.py — the desktop in a 2D mode, placed in space by the glasses' tracking.

With SPACE MODE on in a 2D mode, X stops driving the glasses directly. As in 3D,
braindance's stage sits in X's composite overlay window and draws the desktop
itself, but as one picture for both eyes, at the mode's full rate (up to 120
Hz). Each frame the desktop is composited into a texture, with braindance's
cursor in it, and that texture is drawn as a screen where tracking.py puts it,
from the head pose read just before drawing: flat, or curved around you like a
curved monitor. The settings panel stays fixed in view, over it.

The desktop's size is kept apart from the display's, so a desktop larger than
the view can be placed the same way.
"""
import time

import moderngl
import numpy as np

from . import panel
from .capture import DesktopCapture
from .glasses.pose import PoseReader
from .render import COMPOSITE_FS, COMPOSITE_VS, CURSOR_FS, CURSOR_TRI, CURSOR_VS
from .tracking import Tracker, projection, screen_size

PANEL_APPEAR = 0.30            # the panel's materialize animation, seconds
CURSOR = 0.010                 # cursor half size (outline 1.3x), NDC of a 1920-wide picture

# The desktop on its screen: a strip of COLUMNS quads, flat, or bent onto a
# cylinder around a vertical axis `radius` metres toward the viewer, the
# desktop laid along its arc so every column keeps its width. The outermost
# half pixel fades out, so the screen's edge stays smooth at any angle; flat,
# at one desktop pixel per display pixel, every pixel centre is untouched.
COLUMNS = 256
PLACE_VS = """
#version 330
in vec2 in_uv;
uniform mat4 mvp;         // screen space (metres, centred, facing +z) -> clip
uniform vec2 size;        // the screen's width along its arc, and its height (m)
uniform float radius;     // the curve's radius (m); 0 = flat
out vec2 v_uv;
void main() {
    vec3 p = vec3((in_uv - 0.5) * size, 0.0);
    if (radius > 0.0) {
        float a = p.x / radius;
        p.xz = radius * vec2(sin(a), 1.0 - cos(a));
    }
    gl_Position = mvp * vec4(p, 1.0);
    v_uv = in_uv;
}
"""
PLACE_FS = """
#version 330
in vec2 v_uv;
uniform sampler2D desk;
out vec4 f;
void main() {
    vec2 h = 0.5 * fwidth(v_uv);
    vec2 e = smoothstep(-h, h, v_uv) * smoothstep(-h, h, 1.0 - v_uv);
    f = vec4(texture(desk, v_uv).rgb * e.x * e.y, 1.0);
}
"""


class Placement:
    """Draws a desktop texture as a screen in front of the eyes."""

    def __init__(self, ctx, desk: tuple[int, int], display: tuple[int, int]):
        self.ctx, self.desk, self.display = ctx, desk, display
        self.proj = projection(*display)
        self.prog = ctx.program(vertex_shader=PLACE_VS, fragment_shader=PLACE_FS)
        u = np.linspace(0.0, 1.0, COLUMNS + 1, dtype="f4")
        strip = np.stack([np.repeat(u, 2), np.tile(np.array([0.0, 1.0], "f4"), COLUMNS + 1)], 1)
        self.vao = ctx.vertex_array(self.prog, [(ctx.buffer(strip.tobytes()), "2f", "in_uv")])

    def draw(self, tex, view: np.ndarray, size: float, distance: float, curve: float) -> None:
        """Clear the current framebuffer and draw `tex` as the screen, `view`
        taking screen space to eye space (tracking.Tracker.view). `curve` 1
        bends it around the eyes it was placed for (radius `distance`), 0
        leaves it flat."""
        w, h = screen_size(*self.desk, size, distance)
        self.ctx.viewport = (0, 0, *self.display)
        self.ctx.clear(0.0, 0.0, 0.0)
        tex.use(0)
        self.prog["mvp"].write((self.proj @ view).T.astype("f4").tobytes())
        self.prog["size"].value = (w, h)
        self.prog["radius"].value = distance / curve if curve > 0.0 else 0.0
        self.vao.render(moderngl.TRIANGLE_STRIP)


class Space:
    def __init__(self, stage, mode, S, pose_path: str, six_dof: bool):
        """Take over `stage` (sized for `mode`, a 2D Mode) and capture the desktop
        into it; the head pose comes from the helper's stream at `pose_path`."""
        self.S, self.stage, self.six_dof = S, stage, six_dof
        ctx = self.ctx = stage.ctx
        self.display = (mode.width, mode.height)
        self.desk = mode.desktop
        self.cap = DesktopCapture(ctx, stage.x_display, stage.xid, (0, 0, *self.desk))
        self.pose = PoseReader(pose_path)
        self.tracker = Tracker()

        # the composited desktop, mipmapped: the screen is often drawn smaller
        self.work_tex = ctx.texture(self.desk, 4)
        self.work_tex.repeat_x = self.work_tex.repeat_y = False
        self.work_tex.anisotropy = 16.0
        self.work = ctx.framebuffer(color_attachments=[self.work_tex])
        unit = np.array([0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1], dtype="f4")
        self.comp = ctx.program(vertex_shader=COMPOSITE_VS, fragment_shader=COMPOSITE_FS)
        self.comp["region"].value = tuple(float(v) for v in self.desk)
        self.comp_vao = ctx.vertex_array(self.comp, [(ctx.buffer(unit.tobytes()), "2f", "in_uv")])
        self.cursor = ctx.program(vertex_shader=CURSOR_VS, fragment_shader=CURSOR_FS)
        self.cursor_vao = ctx.vertex_array(
            self.cursor, [(ctx.buffer(CURSOR_TRI.tobytes()), "2f", "in_p")])
        self.placement = Placement(ctx, self.desk, self.display)
        self.panel = panel.Panel(ctx)

        self.fps = 0.0
        self._fps_t, self._fps_n = time.monotonic(), 0
        self._t0 = time.monotonic()

    # --- the scene, worked from beside the panel ------------------------------

    def zoom(self, steps: int) -> None:
        self.S.space_size = round(min(2.0, max(0.3, self.S.space_size * 1.05 ** steps)), 3)

    def pan(self, dx: float, dy: float) -> None:
        self.tracker.nudge(dx, dy)

    def reset_view(self) -> None:
        self.tracker.recenter()

    @property
    def state(self) -> str:
        """For the panel's footer and `braindance status`."""
        return ("6DOF" if self.six_dof else "3DOF") if self.tracker.steady else "WAIT"

    # --- a frame -----------------------------------------------------------------

    def _arrow(self, size, center, color) -> None:
        """The cursor at `center` (NDC) of a picture `size` pixels big."""
        hx = CURSOR * 1920 / size[0]
        hy = hx * size[0] / size[1]
        self.cursor["center"].value = center
        for half, col in ((1.3, (0.0, 0.0, 0.0)), (1.0, color)):   # outline, then fill
            self.cursor["col"].value = col
            self.cursor["halfsz"].value = (hx * half, hy * half)
            self.cursor_vao.render(moderngl.TRIANGLES)

    def frame(self, menu=None, opened_at=0.0, pointer=None, panel_xy=(0, 0)) -> None:
        """Draw one frame, with the settings panel fixed in view when `menu` is
        given: its top-left at `panel_xy` (display pixels), opened at
        `opened_at`, and the cursor on it at `pointer`. Blocks on vsync."""
        S, ctx = self.S, self.ctx
        windows = self.cap.update()
        self.work.use()
        ctx.clear(0.07, 0.07, 0.09)
        for tex, x, y, w, h in windows:
            tex.use(0)
            self.comp["rect"].value = (float(x), float(y), float(x + w), float(y + h))
            self.comp_vao.render(moderngl.TRIANGLES)
        if menu is None:
            u, v, inside = self.cap.pointer()
            if inside:
                self._arrow(self.desk, (u * 2 - 1, (1 - v) * 2 - 1), (0.42, 0.0, 0.55))
        self.work_tex.build_mipmaps()

        # the head pose as late as possible: just before the screen is drawn
        now = time.monotonic()
        view = self.tracker.view(now, self.pose.read(), S.space_mode, self.six_dof,
                                 S.space_distance, S.space_follow)
        ctx.screen.use()
        self.placement.draw(self.work_tex, view, S.space_size, S.space_distance, S.space_curve)

        if menu is not None:
            names, ti = menu.tab_bar()
            self.panel.update(names, ti, menu.rows(), self.fps, ("TRACK", self.state))
            self.panel.draw(panel.ndc(self.display, panel_xy), now - self._t0,
                            min(1.0, (now - opened_at) / PANEL_APPEAR))
            if pointer is not None:
                dw, dh = self.display
                self._arrow(self.display, (pointer[0] / dw * 2 - 1, 1 - pointer[1] / dh * 2),
                            (1.0, 0.16, 0.43))

        self._fps_n += 1
        if now - self._fps_t >= 1.0:
            self.fps = self._fps_n / (now - self._fps_t)
            self._fps_t, self._fps_n = now, 0
        self.stage.swap()
        self.stage.poll()

    def close(self) -> None:
        self.cap.shutdown()             # X paints the desktop again
        self.pose.close()
