"""media.py — a 3D video on the glasses.

libmpv decodes the file (NVDEC when available), keeps the audio in sync, and
renders each new frame into a texture in the stage's GL context. Each eye then
shows its part of the frame, fitted into the eye. mpv's own OSD, OSC and input
bindings are off: mpv would draw them once across the whole side-by-side
frame, so the player draws its own controls into both eyes, and takes the
keyboard and mouse itself while it plays (input.Controls.want).

Layouts: full and half side-by-side, full and half top-bottom, or 2D; named by
the file (iw3's tags, then common player tags), else guessed from the frame's
shape. M cycles them, S swaps the eyes.

  Space, K, left click    pause / play
  Left / Right            seek 5 s (Shift: 30 s)
  PgUp / PgDn, Home       seek 5 min, back to the start
  Up / Down, wheel        volume
  [ / ]                   eye alignment: the picture nearer or farther (Shift: x5)
  - / =                   picture size
  0                       reset alignment and size
  M, S, A                 layout, swap eyes, audio track
  Q, Esc, Backspace       stop
  click the progress bar  seek there
"""
import ctypes
import os
import re
import threading
import time

import glfw
import moderngl
import mpv
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .panel import PAL
from .render import CURSOR_FS, CURSOR_VS

# --- layouts -----------------------------------------------------------------

SBS, HSBS, TB, HTB, FLAT = "sbs", "hsbs", "tb", "htb", "2d"
LAYOUTS = (SBS, HSBS, TB, HTB, FLAT)
LABELS = {SBS: "FULL SBS", HSBS: "HALF SBS", TB: "FULL TB", HTB: "HALF TB", FLAT: "2D"}

# Checked in order against the file name and its folder's. iw3 writes
# _LRF_Full_SBS / _LR / _TBF_fulltb / _TB; the rest are tags other players read.
_TAGS = [
    (re.compile(r"_LRF_Full_SBS|_LRF\b|(?:^|[_.\- ])(?:F|Full)[_\- ]?SBS\b", re.I), SBS),
    (re.compile(r"_TBF_fulltb|_TBF\b|(?:^|[_.\- ])(?:F|Full)[_\- ]?(?:TB|OU)\b", re.I), TB),
    (re.compile(r"_3dh_|_LR\b|(?:^|[_.\- ])(?:H|Half)[_\- ]?SBS\b|(?:^|[_.\- ])SBS\b", re.I), HSBS),
    (re.compile(r"_3dv_|_TB\b|(?:^|[_.\- ])(?:H|Half)[_\- ]?(?:TB|OU)\b|(?:^|[_.\- ])(?:TB|OU)\b",
                re.I), HTB),
]


def layout_from_name(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    folder = os.path.basename(os.path.dirname(path))
    for text in (stem, folder):
        for rx, layout in _TAGS:
            if rx.search(text):
                return layout
    return None


def layout_from_shape(w, h):
    """Full SBS for frames at least 3:1 (two 16:9 eyes are 3.56:1, while scope
    films are 2.39:1), else half SBS: braindance plays 3D files."""
    return SBS if w / h >= 3.0 else HSBS


def eye_aspect(layout, w, h):
    """Display aspect of one eye's picture for a w x h frame."""
    if layout == SBS:
        return (w / 2) / h
    if layout == TB:
        return w / (h / 2)
    return w / h                 # half SBS/TB are squeezed back to the full frame


def eye_region(layout, eye, swap):
    """(u0, v0, u1, v1) of the frame shown to `eye` (0 left, 1 right), with
    (0, 0) at the frame's top-left."""
    if swap:
        eye = 1 - eye
    if layout in (SBS, HSBS):
        return (0.5 * eye, 0.0, 0.5 + 0.5 * eye, 1.0)
    if layout in (TB, HTB):
        return (0.0, 0.5 * eye, 1.0, 0.5 + 0.5 * eye)
    return (0.0, 0.0, 1.0, 1.0)


# --- libmpv into a texture ---------------------------------------------------

def _proc_address(_ctx, name):
    return ctypes.cast(glfw.get_proc_address(name.decode()), ctypes.c_void_p).value


class Video:
    def __init__(self, ctx, volume):
        self.ctx = ctx
        self.mpv = mpv.MPV(
            vo="libmpv", hwdec="auto-safe", keep_open="no", idle="yes",
            config=False, load_scripts=False, osc=False, osd_level=0,
            input_default_bindings=False, input_vo_keyboard=False, ytdl=False,
            sub_visibility=False, keepaspect=False,
            audio_client_name="braindance", volume=volume, volume_max=150)
        self._proc = mpv.MpvGlGetProcAddressFn(_proc_address)
        self.render_ctx = mpv.MpvRenderContext(
            self.mpv, "opengl", opengl_init_params={"get_proc_address": self._proc},
            x11_display=glfw.get_x11_display())
        self._dirty = threading.Event()
        self.render_ctx.update_cb = self._dirty.set
        self.tex = self.fbo = None
        self.size = (0, 0)
        self.has_frame = False

    def load(self, path, start=0.0):
        opts = {"start": f"{start:.2f}"} if start > 1 else {}
        self.mpv.loadfile(path, **opts)

    def prop(self, name, default=None):
        try:
            v = self.mpv._get_property(name)
        except Exception:
            return default
        return default if v is None else v

    def set(self, name, value):
        try:
            self.mpv._set_property(name, value)
        except Exception:
            pass

    def command(self, *args):
        try:
            self.mpv.command(*args)
        except Exception:
            pass

    def update(self):
        """Render mpv's next frame into the texture if it has one."""
        if not self._dirty.is_set():
            return
        self._dirty.clear()
        if not self.render_ctx.update():
            return
        w, h = self.prop("dwidth"), self.prop("dheight")       # sample aspect applied
        if not w or not h:
            return
        if (w, h) != self.size:
            if self.fbo:
                self.fbo.release()
                self.tex.release()
            self.tex = self.ctx.texture((w, h), 4)
            self.tex.repeat_x = self.tex.repeat_y = False
            self.fbo = self.ctx.framebuffer(color_attachments=[self.tex])
            self.size = (w, h)
        # don't wait for the frame's display time here: the stage's vsync'd swap paces us
        self.render_ctx.render(flip_y=False, block_for_target_time=False,
                               opengl_fbo={"w": w, "h": h, "fbo": self.fbo.glo})
        self.has_frame = True

    def report_swap(self):
        self.render_ctx.report_swap()

    def close(self):
        self.render_ctx.free()
        self.mpv.terminate()
        if self.fbo:
            self.fbo.release()
            self.tex.release()


# --- the OSD -----------------------------------------------------------------

OSD_W, OSD_H = 1100, 132


def _font(size, bold=False):
    try:
        return ImageFont.truetype("DejaVuSansMono" + ("-Bold" if bold else "") + ".ttf", size)
    except OSError:
        return ImageFont.load_default()


def _time(t):
    t = int(max(0, t or 0))
    h, m, s = t // 3600, t // 60 % 60, t % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _fit(d, text, font, width):
    if d.textlength(text, font=font) <= width:
        return text
    while text and d.textlength(text + "…", font=font) > width:
        text = text[:-1]
    return text + "…"


BAR_X0, BAR_X1, BAR_Y = 28, OSD_W - 28, 84      # the progress bar, in OSD pixels


def paint_osd(title, pos, dur, paused, volume, layout, swap, depth, size, toast):
    img = Image.new("RGBA", (OSD_W, OSD_H), PAL["bg"])
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, OSD_W - 1, OSD_H - 1], outline=PAL["border"], width=2)
    d.text((28, 20), "❚❚" if paused else "▶", font=_font(30, True), fill=PAL["magenta"])
    right = (f"{LABELS[layout]}{' ⇄' if swap else ''}   ALIGN {depth:+.0f}   "
             f"SIZE {size * 100:.0f}%   VOL {int(volume)}")
    fr = _font(20)
    rw = d.textlength(right, font=fr)
    d.text((OSD_W - 28 - rw, 27), right, font=fr, fill=PAL["cyan"])
    ft = _font(26, True)
    d.text((90, 21), _fit(d, toast or title, ft, OSD_W - 140 - rw), font=ft,
           fill=PAL["yellow"] if toast else PAL["fg_hi"])
    d.rectangle([BAR_X0, BAR_Y, BAR_X1, BAR_Y + 8], fill=(40, 28, 70, 255))
    if dur:
        fx = BAR_X0 + int((BAR_X1 - BAR_X0) * max(0.0, min(1.0, pos / dur)))
        d.rectangle([BAR_X0, BAR_Y, fx, BAR_Y + 8], fill=PAL["magenta"])
        d.rectangle([fx - 3, BAR_Y - 6, fx + 3, BAR_Y + 14], fill=PAL["fg_hi"])
    fs = _font(18)
    d.text((BAR_X0, BAR_Y + 18), _time(pos), font=fs, fill=PAL["fg"])
    if dur:
        rem = f"-{_time(dur - pos)} / {_time(dur)}"
        d.text((BAR_X1 - d.textlength(rem, font=fs), BAR_Y + 18), rem, font=fs, fill=PAL["fg"])
    return img


# --- shaders -------------------------------------------------------------------

# One eye's picture: a rect of the frame texture, onto a rect of the eye (NDC).
EYE_VS = """
#version 330
in vec2 in_uv;              // unit quad, v=0 at the top
uniform vec4 rect;          // NDC x0, y0, x1, y1 (y up)
out vec2 v_uv;
void main() {
    gl_Position = vec4(mix(rect.x, rect.z, in_uv.x), mix(rect.w, rect.y, in_uv.y), 0.0, 1.0);
    v_uv = in_uv;
}
"""
EYE_FS = """
#version 330
in vec2 v_uv;
uniform sampler2D tex;
uniform vec4 region;        // u0, v0, u1, v1 of the frame, (0, 0) = its top-left
out vec4 f;
void main() {
    // mpv renders the frame upright (flip_y off): its top is the texture's row 0
    f = vec4(texture(tex, mix(region.xy, region.zw, v_uv)).rgb, 1.0);
}
"""
OVERLAY_FS = """
#version 330
in vec2 v_uv;
uniform sampler2D tex;      // RGBA, row 0 = top
out vec4 f;
void main() { f = texture(tex, v_uv); }
"""

OSD_SECS = 3.0        # the OSD and cursor stay this long after the last input
TOAST_SECS = 1.6
FLOAT_PX = 6          # the OSD and cursor float this far in front of the picture, per eye


class Player:
    def __init__(self, stage, mode, path, S, start=0.0):
        self.ctx = stage.ctx
        self.S = S
        self.path = path
        self.title = os.path.splitext(os.path.basename(path))[0]
        self.eye_w, self.eye_h = mode.desktop
        self.video = Video(self.ctx, volume=S.media_volume)
        self.video.load(path, start)
        self.layout = layout_from_name(path)        # None until the first frame's shape
        self.swap = False
        self.done = False
        self.error = None
        unit = np.array([0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1], dtype="f4")
        self.eye = self.ctx.program(vertex_shader=EYE_VS, fragment_shader=EYE_FS)
        self.eye_vao = self.ctx.vertex_array(self.eye, [(self.ctx.buffer(unit.tobytes()), "2f", "in_uv")])
        self.overlay = self.ctx.program(vertex_shader=EYE_VS, fragment_shader=OVERLAY_FS)
        self.overlay_vao = self.ctx.vertex_array(
            self.overlay, [(self.ctx.buffer(unit.tobytes()), "2f", "in_uv")])
        self.osd_tex = self.ctx.texture((OSD_W, OSD_H), 4)
        self.osd_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._osd_key = None
        tri = np.array([0.0, 0.0, 0.0, -1.4, 0.95, -0.95], dtype="f4")
        self.cursor = self.ctx.program(vertex_shader=CURSOR_VS, fragment_shader=CURSOR_FS)
        self.cursor_vao = self.ctx.vertex_array(self.cursor, [(self.ctx.buffer(tri.tobytes()), "2f", "in_p")])
        now = time.monotonic()
        self.shown_until = now + OSD_SECS
        self.toast, self.toast_until = None, 0.0
        self.pointer = (self.eye_w / 2, self.eye_h / 2)
        self._started = now

    # --- state -----------------------------------------------------------------

    def _note(self, text):
        now = time.monotonic()
        self.toast, self.toast_until = text, now + TOAST_SECS
        self.shown_until = max(self.shown_until, now + TOAST_SECS)

    def position(self):
        return self.video.prop("time-pos", 0.0)

    def _check_done(self):
        if self.video.prop("idle-active", False):
            if not self.video.has_frame and time.monotonic() - self._started > 0.5:
                self.error = "mpv could not play it"
            self.done = self.video.has_frame or self.error is not None

    # --- input -------------------------------------------------------------------

    def key(self, name, shift=False):
        v, S = self.video, self.S
        self.shown_until = time.monotonic() + OSD_SECS
        if name in ("space", "k"):
            v.command("cycle", "pause")
        elif name in ("left", "right"):
            step = 30 if shift else 5
            v.command("seek", str(step if name == "right" else -step), "relative+exact")
        elif name in ("pageup", "pagedown"):
            v.command("seek", "300" if name == "pagedown" else "-300", "relative")
        elif name == "home":
            v.command("seek", "0", "absolute")
        elif name in ("up", "down"):
            self._volume(5 if name == "up" else -5)
        elif name in ("bracketleft", "bracketright"):
            step = 10 if shift else 2
            S.media_depth = max(-80.0, min(160.0, S.media_depth
                                           + (step if name == "bracketright" else -step)))
            self._note(f"ALIGN {S.media_depth:+.0f}")
        elif name in ("minus", "equal"):
            S.media_size = round(max(0.3, min(1.5, S.media_size
                                              + (0.05 if name == "equal" else -0.05))), 2)
            self._note(f"SIZE {S.media_size * 100:.0f}%")
        elif name == "0":
            S.media_depth, S.media_size = 0.0, 1.0
            self._note("ALIGN AND SIZE RESET")
        elif name == "m" and self.layout:
            self.layout = LAYOUTS[(LAYOUTS.index(self.layout) + 1) % len(LAYOUTS)]
            self._note(LABELS[self.layout])
        elif name == "s":
            self.swap = not self.swap
            self._note("EYES SWAPPED" if self.swap else "EYES NORMAL")
        elif name == "a":
            v.command("cycle", "aid")
            self._note(f"AUDIO TRACK {v.prop('aid') or 'OFF'}")
        elif name in ("q", "escape", "backspace"):
            self.done = True

    def _volume(self, delta):
        vol = max(0, min(150, int(self.video.prop("volume", self.S.media_volume)) + delta))
        self.video.set("volume", vol)
        self.S.media_volume = vol

    def _osd_rect_px(self):
        """The OSD's rect in eye pixels (x0, y0, x1, y1), y down."""
        x0 = (self.eye_w - OSD_W) / 2
        y1 = self.eye_h - 40
        return (x0, y1 - OSD_H, x0 + OSD_W, y1)

    def mouse(self, kind, x, y, button=0, shift=False):
        """kind: "move", "press" or "release"; x, y in eye pixels (the desktop's)."""
        self.pointer = (x, y)
        now = time.monotonic()
        visible = now < self.shown_until
        self.shown_until = now + OSD_SECS
        if kind != "press":
            return
        if button == 1:
            ox0, oy0, ox1, _ = self._osd_rect_px()
            dur = self.video.prop("duration", 0.0)
            bx0, bx1 = ox0 + BAR_X0, ox0 + BAR_X1
            if (visible and dur and bx0 - 10 <= x <= bx1 + 10
                    and abs(y - (oy0 + BAR_Y + 4)) <= 22):
                frac = max(0.0, min(1.0, (x - bx0) / (bx1 - bx0)))
                self.video.command("seek", f"{frac * dur:.2f}", "absolute")
            else:
                self.video.command("cycle", "pause")
        elif button in (4, 5):
            self._volume(5 if button == 4 else -5)

    # --- drawing -----------------------------------------------------------------

    def _ndc(self, x0, y0, x1, y1, shift):
        """Eye-pixel rect (y down) shifted `shift` px right -> NDC rect (y up)."""
        ew, eh = self.eye_w, self.eye_h
        return ((x0 + shift) / ew * 2 - 1, 1 - y1 / eh * 2,
                (x1 + shift) / ew * 2 - 1, 1 - y0 / eh * 2)

    def draw(self, ui=True):
        """Draw both eyes into the current framebuffer (the stage, 2 x eye wide),
        with the OSD and cursor unless `ui` is off (the settings panel is open
        over the video). Returns False once playback is over."""
        v, S, ctx = self.video, self.S, self.ctx
        v.update()
        self._check_done()
        ctx.viewport = (0, 0, 2 * self.eye_w, self.eye_h)
        ctx.clear(0.0, 0.0, 0.0)
        now = time.monotonic()
        show = ui and (now < self.shown_until or bool(v.prop("pause", False)))
        if v.has_frame:
            w, h = v.size
            if self.layout is None:
                self.layout = layout_from_shape(w, h)
            ar = eye_aspect(self.layout, w, h)
            ew, eh = self.eye_w, self.eye_h
            pw = ew * S.media_size
            ph = pw / ar
            if ph > eh * S.media_size:
                ph = eh * S.media_size
                pw = ph * ar
            x0, y0 = (ew - pw) / 2, (eh - ph) / 2
            v.tex.use(0)
            for half in range(2):
                sign = 1.0 if half == 0 else -1.0     # + = nearer: left eye right, right eye left
                ctx.viewport = (half * ew, 0, ew, eh)
                self.eye["rect"].value = self._ndc(x0, y0, x0 + pw, y0 + ph, sign * S.media_depth / 2)
                self.eye["region"].value = eye_region(self.layout, half, self.swap)
                self.eye_vao.render(moderngl.TRIANGLES)
        if show:
            self._draw_osd(now)
        return not self.done

    def _draw_osd(self, now):
        v, S, ctx = self.video, self.S, self.ctx
        toast = self.toast if now < self.toast_until else None
        pos, dur = v.prop("time-pos", 0.0), v.prop("duration", 0.0)
        key = (int(pos), int(dur), bool(v.prop("pause", False)), int(v.prop("volume", 0)),
               self.layout, self.swap, S.media_depth, S.media_size, toast)
        if key != self._osd_key:
            self._osd_key = key
            img = paint_osd(self.title, pos, dur, key[2], key[3], self.layout or SBS,
                            self.swap, S.media_depth, S.media_size, toast)
            self.osd_tex.write(img.tobytes())
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        rx0, ry0, rx1, ry1 = self._osd_rect_px()
        px, py = self.pointer
        ew, eh = self.eye_w, self.eye_h
        asp = ew / eh
        for half in range(2):
            sign = 1.0 if half == 0 else -1.0
            shift = sign * (S.media_depth / 2 + FLOAT_PX)
            ctx.viewport = (half * ew, 0, ew, eh)
            self.osd_tex.use(0)
            self.overlay["rect"].value = self._ndc(rx0, ry0, rx1, ry1, shift)
            self.overlay_vao.render(moderngl.TRIANGLES)
            cx, cy = (px + shift) / ew * 2 - 1, 1 - py / eh * 2
            self.cursor["center"].value = (cx, cy)
            self.cursor["col"].value = (0.0, 0.0, 0.0)
            self.cursor["halfsz"].value = (0.013, 0.013 * asp)
            self.cursor_vao.render(moderngl.TRIANGLES)
            self.cursor["col"].value = (0.42, 0.0, 0.55)
            self.cursor["halfsz"].value = (0.010, 0.010 * asp)
            self.cursor_vao.render(moderngl.TRIANGLES)
        ctx.disable(moderngl.BLEND)

    def report_swap(self):
        self.video.report_swap()

    def close(self):
        self.S.media_volume = int(self.video.prop("volume", self.S.media_volume))
        self.video.close()
        for o in (self.osd_tex, self.eye_vao, self.eye, self.overlay_vao, self.overlay,
                  self.cursor_vao, self.cursor):
            o.release()
