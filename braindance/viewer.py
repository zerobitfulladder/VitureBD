"""viewer.py — a 3D model (glTF / GLB / VRM) or a Gaussian splat scene (.splat,
3DGS .ply) on the glasses, placed in the room.

`braindance view FILE` opens it on the stage as a video plays (session.py):
in a 3D mode, switching from a 2D one for it and back after. The model is
drawn in stereo, one view per eye 63 mm apart, from the head pose read just
before each frame (glasses/pose.py), so it stays where it was placed (ANCHOR)
or glides after the head (FOLLOW): tracking.py, upright, a model's base level
whatever the head's tilt. Everything but the model is black, which the
glasses' OLED panels show as clear. model.py loads and shades a model, and
splats.py a splat scene.

  left or middle drag     turn the model (arrows: by 15 degrees)
  Shift + middle drag      move it left, right, up and down (as in Blender)
  wheel, - / =             size
  middle click, R          recenter (and the mouse's back side button)
  0                        undo the turn, move and size
  Q, Esc, Backspace        close
"""
import math
import os
import time

import numpy as np

from .glasses.pose import PoseReader
from .model import Model
from .splats import SPLAT_TYPES, Splats, is_splats
from .settings import DEFAULTS
from .tracking import DISPLAY_W, TAN_HALF_W, Tracker, projection, translate

MODEL_TYPES = (".glb", ".gltf", ".vrm") + SPLAT_TYPES
IPD = 0.063                    # metres between the eyes' views
DRAG = 0.4                     # degrees of turn per pixel dragged
CLICK = 3                      # a middle press that moves less (px) is a click: recenter
SAMPLES = 4                    # a model's multisampling: smooth edges against the black


def is_model(path: str) -> bool:
    """A 3D model or splat scene for the viewer (else a video for the player)."""
    return path.lower().endswith(MODEL_TYPES)


def _turn(yaw: float, pitch: float) -> np.ndarray:
    """Turn about the vertical by `yaw`, then tip about the horizontal by
    `pitch` (degrees), as a 4x4."""
    cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    m = np.eye(4)
    m[:3, :3] = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]]) @ \
        np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return m


class ModelView:
    def __init__(self, stage, mode, path, S, pose_path: str, six_dof: bool):
        self.ctx = ctx = stage.ctx
        self.S, self.path, self.six_dof = S, path, six_dof
        self.title = os.path.splitext(os.path.basename(path))[0]
        self.eye_w, self.eye_h = mode.desktop
        self.proj = projection(self.eye_w, self.eye_h)
        self.error = None
        self.splats = is_splats(path)
        try:
            self.model = (Splats if self.splats else Model)(ctx, path)
        except Exception as e:
            self.model, self.error = None, f"cannot load it: {e}"
        self.done = self.error is not None
        self.pose = PoseReader(pose_path)
        self.tracker = Tracker()
        self.msaa = None              # splats are soft already; a model's edges are not
        if not self.splats:
            size = (2 * self.eye_w, self.eye_h)
            self.msaa = ctx.framebuffer(
                color_attachments=[ctx.renderbuffer(size, 4, samples=SAMPLES)],
                depth_attachment=ctx.depth_renderbuffer(size, samples=SAMPLES))
        self.yaw = self.pitch = 0.0
        self.move = np.zeros(2)       # metres right and up, in front of the placement
        self._drag = None             # [what, button, x, y, px moved]

    # --- input --------------------------------------------------------------------

    def recenter(self) -> None:
        self.tracker.recenter()
        self.move[:] = 0.0

    def _size(self, steps: int) -> None:
        self.S.model_size = round(min(5.0, max(0.05, self.S.model_size * 1.1 ** steps)), 3)

    def key(self, name, shift=False):
        if name in ("q", "escape", "backspace"):
            self.done = True
        elif name == "r":
            self.recenter()
        elif name in ("minus", "equal"):
            self._size(1 if name == "equal" else -1)
        elif name in ("left", "right"):
            self.yaw += 15.0 if name == "right" else -15.0
        elif name in ("up", "down"):
            self.pitch += 15.0 if name == "down" else -15.0
        elif name == "0":
            self.yaw = self.pitch = 0.0
            self.move[:] = 0.0
            self.S.model_size = DEFAULTS["model_size"]

    def mouse(self, kind, x, y, button=0, shift=False):
        """kind: "press", "release" or "move"; x, y in root pixels; `shift` held
        at a press."""
        if kind == "press":
            if button in (1, 2):
                self._drag = ["move" if button == 2 and shift else "turn", button, x, y, 0]
            elif button in (4, 5):
                self._size(1 if button == 4 else -1)
        elif kind == "release" and self._drag is not None and button == self._drag[1]:
            if button == 2 and self._drag[4] < CLICK:
                self.recenter()
            self._drag = None
        elif kind == "move" and self._drag is not None:
            what, _, x0, y0, moved = self._drag
            dx, dy = x - x0, y - y0
            if what == "turn":
                self.yaw += dx * DRAG
                self.pitch = max(-90.0, min(90.0, self.pitch + dy * DRAG))
            else:                     # with the pointer, at the model's distance
                k = self.S.model_distance * 2 * TAN_HALF_W / DISPLAY_W
                self.move += (dx * k, -dy * k)
            self._drag[2:] = [x, y, moved + abs(dx) + abs(dy)]

    def position(self) -> float:
        return 0.0

    # --- drawing -----------------------------------------------------------------

    def draw(self, ui=True):
        """Draw both eyes into the current framebuffer (the stage, 2 x eye wide).
        Returns False once closed."""
        ctx, S = self.ctx, self.S
        target = ctx.fbo
        if self.model is None:
            ctx.clear(0.0, 0.0, 0.0)
            return False
        place = self.tracker.view(time.monotonic(), self.pose.read(), S.model_mode, self.six_dof,
                                  S.model_distance, S.model_follow, upright=True)
        size = S.model_size
        model = (place @ translate(*self.move, 0.0) @ _turn(self.yaw, self.pitch)
                 @ np.diag([size, size, size, 1.0]))
        if self.splats:
            self.model.sort(model)    # back to front, once for both eyes
        (target if self.splats else self.msaa).use()
        ctx.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)
        for half in range(2):
            ctx.viewport = (half * self.eye_w, 0, self.eye_w, self.eye_h)
            eye = translate(IPD / 2 if half == 0 else -IPD / 2, 0.0, 0.0)  # head -> this eye
            self.model.draw(self.proj, eye @ model, place[:3, :3], S.model_exposure)
        if not self.splats:
            ctx.copy_framebuffer(target, self.msaa)
            target.use()
        return not self.done

    def report_swap(self):
        pass

    def close(self):
        if self.model is not None:
            self.model.release()
        if self.msaa is not None:
            for o in (*self.msaa.color_attachments, self.msaa.depth_attachment, self.msaa):
                o.release()
        self.pose.close()
