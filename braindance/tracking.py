"""tracking.py — where the screen sits, from where the head is.

The glasses report the head's pose (glasses/pose.py) in OpenGL axes: x right,
y up, z backward, in metres, with a quaternion that turns head axes into world
axes. With 3DOF there is no position: the head stays at the origin.

The screen's centre is `distance` metres in front of the eyes when it is
placed (space.Placement may bend it around them). Its size is set in desktop
pixels: at SIZE 1 one desktop pixel spans one display pixel at the screen's
centre, so a desktop larger than the display would simply reach past the
edges of the view.

  ANCHOR  The screen stays where it was placed, square to your view as it was
          then: placed with the head's roll too, not level with gravity, so it
          lines up with the display however the glasses sit. With 3DOF leaning
          carries it along; with 6DOF it stays in the room and you can lean
          in. A one-euro filter holds the head pose still against sensor
          noise, so text doesn't shimmer, and lets real motion through.
  FOLLOW  The screen always glides after the head, however slowly it moves:
          it trails a turn by the FOLLOW time constant and settles in front
          of you once you stop.

Something upright (a 3D model, viewer.py) is placed the same way, along the
gaze, but turned by the head's heading alone, so it stands level whatever the
head's pitch and roll.

recenter() places the screen in front of you again; nudge() moves an anchored
screen by a drag.
"""
import math

import numpy as np

# The Luma Ultra's view, from the display frustum in the glasses' factory
# calibration: +-0.02 x +-0.01125 at 0.0508 m, 43 x 25 degrees over 1920 x 1080
# (a camera calibration of these glasses found 25.03 degrees). Its pixels are
# square, so the 1200-row modes see a little taller. VITURE's "52 degrees" is
# wider than this, and anchoring with it let the screen drift with the head.
DISPLAY_W = 1920
TAN_HALF_W = 0.02 / 0.0508

STALE = 0.25          # a pose sample older than this (s) is not used
# the one-euro filter on the head pose in ANCHOR: cutoff (Hz) at rest, and its
# rise with speed (Hz per deg/s, per m/s)
ROT_CUTOFF, ROT_BETA = 1.0, 3.0
POS_CUTOFF, POS_BETA = 1.0, 30.0
D_CUTOFF = 1.0        # cutoff for the speed estimate itself


# --- quaternions (w, x, y, z) and 4x4 matrices -----------------------------

def _qnorm(q):
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q)


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _axis(axis, angle):
    s = math.sin(angle / 2)
    return np.array([math.cos(angle / 2), axis[0] * s, axis[1] * s, axis[2] * s])


def _qmat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _qangle(a, b) -> float:
    """The angle between two orientations, degrees."""
    return math.degrees(2 * math.acos(min(1.0, abs(float(np.dot(a, b))))))


def _slerp(a, b, t):
    d = float(np.dot(a, b))
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        return _qnorm(a + (b - a) * t)
    th = math.acos(d)
    return (math.sin((1 - t) * th) * a + math.sin(t * th) * b) / math.sin(th)


def _pose(q, p):
    """The matrix of a frame at position p with orientation q."""
    m = np.eye(4)
    m[:3, :3] = _qmat(q)
    m[:3, 3] = p
    return m


def _inverse(q, p):
    r = _qmat(q).T
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = -r @ p
    return m


def _heading(q):
    """The turn about the vertical alone of an orientation."""
    f = _qmat(q) @ (0.0, 0.0, -1.0)
    return _axis((0, 1, 0), math.atan2(-f[0], -f[2]))


def _placed(q, p, distance, upright):
    """Where a thing placed for a head at (q, p) sits: `distance` ahead along its
    gaze, facing it, turned like the head or, upright, by its heading alone."""
    return _pose(_heading(q) if upright else q, p + _qmat(q) @ (0.0, 0.0, -distance))


def translate(x, y, z):
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


def projection(display_w: int, display_h: int, near=0.05, far=100.0):
    """The glasses' view as a GL projection, for a display of that many pixels."""
    tw = TAN_HALF_W * display_w / DISPLAY_W
    th = TAN_HALF_W * display_h / DISPLAY_W
    return np.array([[1 / tw, 0, 0, 0],
                     [0, 1 / th, 0, 0],
                     [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
                     [0, 0, -1, 0]])


def screen_size(desk_w: int, desk_h: int, size: float, distance: float) -> tuple[float, float]:
    """The screen's width and height in metres: at size 1 a desktop pixel spans
    a display pixel at its centre."""
    w = 2 * distance * TAN_HALF_W * size * desk_w / DISPLAY_W
    return w, w * desk_h / desk_w


# --- filtering -----------------------------------------------------------------

def _alpha(dt: float, cutoff: float) -> float:
    return 1.0 / (1.0 + 1.0 / (2 * math.pi * cutoff * dt))


class _OneEuro:
    """Casiez et al.'s one-euro filter: a low-pass whose cutoff rises with
    speed, so it smooths noise at rest and adds almost no lag to motion."""

    def __init__(self, cutoff, beta, dist, blend):
        self.cutoff, self.beta, self.dist, self.blend = cutoff, beta, dist, blend
        self.x = self.raw = None
        self.speed = 0.0

    def __call__(self, dt: float, x):
        if self.x is None:
            self.x = self.raw = x
            return x
        if dt > 0:
            self.speed += _alpha(dt, D_CUTOFF) * (self.dist(self.raw, x) / dt - self.speed)
            self.x = self.blend(self.x, x, _alpha(dt, self.cutoff + self.beta * self.speed))
            self.raw = x
        return self.x


# --- the tracker -------------------------------------------------------------------

class Tracker:
    def __init__(self):
        self._mode = None
        self._recenter = True
        self._last_t = None
        self._rot = _OneEuro(ROT_CUTOFF, ROT_BETA, _qangle, _slerp)
        self._pos = _OneEuro(POS_CUTOFF, POS_BETA, lambda a, b: float(np.linalg.norm(b - a)),
                             lambda a, b, t: a + (b - a) * t)
        self._anchor = None          # ANCHOR: the head's (orientation, position) when placed
        self._follow = None          # FOLLOW: (orientation, position) the screen rides on
        self.steady = False          # the last view() had a steady pose to use

    def recenter(self) -> None:
        self._recenter = True

    def nudge(self, dx: float, dy: float) -> None:
        """Move an anchored screen by a drag of (dx, dy) display pixels."""
        if self._anchor is None or self._mode != "anchor":
            return
        k = 2 * TAN_HALF_W / DISPLAY_W           # radians per pixel at the centre
        q, p = self._anchor                      # turn about the vertical, tilt about its own x
        self._anchor = (_qmul(_axis((0, 1, 0), -dx * k), _qmul(q, _axis((1, 0, 0), -dy * k))), p)

    def view(self, now: float, sample, mode: str, six_dof: bool, distance: float,
             follow: float, upright: bool = False) -> np.ndarray:
        """The matrix from screen space (metres, centred, facing +z) to eye space
        for this frame. Without a steady, recent pose: the screen fixed in view."""
        ahead = translate(0, 0, -distance)
        self.steady = (sample is not None and now - sample.t < STALE and sample.status == 0)
        if not self.steady:
            return ahead
        if mode != self._mode:
            self._mode, self._recenter = mode, True
        q = _qnorm(sample.quat)
        p = np.array(sample.pos, dtype=np.float64) if six_dof else np.zeros(3)
        dt = 0.0 if self._last_t is None else sample.t - self._last_t
        self._last_t = sample.t
        if mode == "anchor":
            return self._anchored(dt, q, p, distance, upright)
        return self._following(dt, q, p, distance, follow, upright)

    def _anchored(self, dt, q, p, distance, upright):
        q, p = self._rot(dt, q), self._pos(dt, p)
        if self._recenter or self._anchor is None:
            self._anchor, self._recenter = (q, p), False
        return _inverse(q, p) @ _placed(*self._anchor, distance, upright)

    def _following(self, dt, q, p, distance, follow, upright):
        if self._recenter or self._follow is None:
            self._follow, self._recenter = (q, p), False
        qs, ps = self._follow
        a = 1.0 - math.exp(-dt / max(follow, 1e-3)) if dt > 0 else 0.0
        self._follow = (_slerp(qs, q, a), ps + (p - ps) * a)
        return _inverse(q, p) @ _placed(*self._follow, distance, upright)
