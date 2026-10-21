"""panel.py — the settings panel: the menu drawn as a texture, shown with an
animated cyberpunk shader.

It draws into whatever surface is current. In a 3D mode the renderer draws it
once per eye, floated in front of the scene; in a 2D mode it fills a small
window of its own (presenter.PanelWindow) above the desktop.
"""
import numpy as np
import moderngl
from PIL import Image, ImageDraw, ImageFont

# A textured quad over an NDC rect, plus a per-eye x offset (disparity).
PANEL_VS = """
#version 330
in vec2 in_uv;         // unit quad, v=0 at the top
uniform vec4 rect;     // NDC x0, y0, x1, y1
uniform vec2 off;
out vec2 v_uv;
void main() {
    vec2 p = vec2(mix(rect.x, rect.z, in_uv.x), mix(rect.w, rect.y, in_uv.y));
    gl_Position = vec4(p + off, 0.0, 1.0);
    v_uv = in_uv;
}
"""

# Samples the cached panel bitmap with cyberpunk effects: chromatic aberration
# (subtle, amplified during a glitch burst), occasional horizontal-slice glitch +
# noise flash, a breathing magenta edge-glow, and a noise "materialize" dissolve
# on open (`appear` 0->1). No scanlines, no flicker. `time` = small seconds.
PANEL_FS = """
#version 330
in vec2 v_uv;
uniform sampler2D tex;
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
    float r = texture(tex, clamp(uv + vec2(ca, 0.0), 0.0, 1.0)).r;
    vec4  m = texture(tex, clamp(uv, 0.0, 1.0));
    float b = texture(tex, clamp(uv - vec2(ca, 0.0), 0.0, 1.0)).b;
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
        vec4  ts   = texture(tex, tuv);
        vec3  tc   = vec3(texture(tex, clamp(tuv + vec2(sp, 0.0), 0.0, 1.0)).r,
                          ts.g,
                          texture(tex, clamp(tuv - vec2(sp, 0.0), 0.0, 1.0)).b);
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

# Night City palette (mirrors ~/Scripts/palette_cyberpunk.py).
PAL = {
    "bg": (10, 4, 24, 232), "border": (255, 42, 109), "magenta": (255, 42, 109),
    "cyan": (5, 217, 232), "yellow": (252, 238, 12), "lime": (204, 255, 0),
    "violet": (185, 103, 255), "fg": (160, 168, 200), "fg_hi": (200, 208, 232),
    "muted": (90, 74, 120), "error": (255, 0, 60),
}

W, H = 640, 496             # the panel bitmap (px): fits the longest tab, 2 sections + 8 rows
FULL = (-1.0, -1.0, 1.0, 1.0)


# Where the panel sits is kept as fractions of the room it can move in: (0, 0)
# puts it in the viewport's top-left corner, (1, 1) in its bottom-right. Being
# fractions, a place holds across the 1080- and 1200-line modes.


def place(view, pos) -> tuple[int, int]:
    """The panel's top-left in pixels (y down) of a view (w, h), for `pos`."""
    fx, fy = (min(1.0, max(0.0, float(v))) for v in pos)
    return round(fx * (view[0] - W)), round(fy * (view[1] - H))


def where(view, xy) -> list[float]:
    """The inverse of place(), for a top-left dragged to `xy` (kept on screen)."""
    return [round(min(1.0, max(0.0, xy[0] / (view[0] - W))), 4),
            round(min(1.0, max(0.0, xy[1] / (view[1] - H))), 4)]


def ndc(view, xy) -> tuple[float, float, float, float]:
    """The NDC rect (x0, y0, x1, y1; y up) of the panel at 1:1 pixel size with its
    top-left at `xy`."""
    w, h = view
    x0, y_top = xy
    return (x0 / w * 2 - 1, 1 - (y_top + H) / h * 2,
            (x0 + W) / w * 2 - 1, 1 - y_top / h * 2)


def hit(hits, x, y):
    """What panel pixel (x, y) is on: ("tab", i), ("row", i), ("panel", None) for
    the rest of the panel, or None outside it. `hits` comes from paint()."""
    if not (0 <= x < W and 0 <= y < H):
        return None
    for x0, y0, x1, y1, kind, i in hits:
        if x0 <= x < x1 and y0 <= y < y1:
            return kind, i
    return "panel", None


def _val_color(row):
    if not row["editable"]:
        return PAL["muted"]
    if row["kind"] == "bool":
        return PAL["lime"] if row["value"] == "ON" else PAL["muted"]
    if row["kind"] == "action":
        return PAL["magenta"]
    if row["kind"] == "enum":
        return PAL["cyan"]
    return PAL["yellow"]


def _fonts():
    try:
        return (ImageFont.truetype("DejaVuSansMono.ttf", 21),
                ImageFont.truetype("DejaVuSansMono-Bold.ttf", 26),
                ImageFont.truetype("DejaVuSansMono.ttf", 16))
    except OSError:
        f = ImageFont.load_default()
        return f, f, f


def paint(tab_names, tab_idx, rows, fps=None, status=None):
    """The panel bitmap, tabbed, single column, and its hit map for hit().
      rows:  the current tab only — section {section:True,label} or
             setting {section:False,label,value,kind,editable,selected}.
      fps:    the display's frame rate for the footer, or None.
      status: (label, value) beside it, as ("DEPTH", "20/s") or ("TRACK", "6DOF"),
              or None.
    Returns (image, hits)."""
    font, font_b, font_s = _fonts()
    hits = []
    img = Image.new("RGBA", (W, H), PAL["bg"])
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W - 1, H - 1], outline=PAL["border"], width=2)

    pad = 28
    # title
    d.text((pad, 12), "BRAINDANCE", font=font_b, fill=PAL["magenta"])
    tw = d.textlength("BRAINDANCE", font=font_b)
    d.text((pad + tw + 14, 18), "v4.0-NC", font=font_s, fill=PAL["yellow"])

    # tab strip: the tabs' side padding shrinks so they all fit on one line
    tx, ty, gap = pad, 46, 5
    widths = [d.textlength(name, font=font) for name in tab_names]
    side = min(22.0, (W - 2 * pad - gap * (len(widths) - 1) - sum(widths)) / len(widths))
    for i, (name, tw) in enumerate(zip(tab_names, widths)):
        w = tw + side
        hits.append((tx, ty, tx + w + gap, ty + 32, "tab", i))
        if i == tab_idx:
            d.rectangle([tx, ty, tx + w, ty + 30], fill=(36, 24, 64, 255))
            d.rectangle([tx, ty + 28, tx + w, ty + 30], fill=PAL["magenta"])
            d.text((tx + side / 2, ty + 4), name, font=font, fill=PAL["fg_hi"])
        else:
            d.text((tx + side / 2, ty + 4), name, font=font, fill=PAL["muted"])
        tx += w + gap
    d.line([pad, ty + 36, W - pad, ty + 36], fill=PAL["muted"], width=1)

    # settings column (single, full panel width)
    cw, lh = W - 2 * pad, 34
    y = ty + 48
    for ri, r in enumerate(rows):
        if r.get("section"):
            y += 6
            d.rectangle([pad, y, pad + cw, y + 24], fill=(26, 16, 48, 255))
            d.rectangle([pad, y, pad + 5, y + 24], fill=PAL["cyan"])
            d.text((pad + 14, y + 1), r["label"], font=font_s, fill=PAL["cyan"])
            y += 24 + 6
            continue
        hits.append((pad, y - 4, pad + cw, y + 30, "row", ri))
        sel = r["selected"]
        if sel:
            d.rectangle([pad, y - 4, pad + cw, y + 26], fill=(36, 24, 64, 255))
            d.rectangle([pad, y - 4, pad + 5, y + 26], fill=PAL["magenta"])
        d.text((pad + 16, y), r["label"], font=font,
               fill=PAL["fg_hi"] if sel else PAL["fg"])
        val = r["value"]
        col = _val_color(r)
        vx = pad + cw - 190
        if sel and r["editable"] and r["kind"] != "action":
            d.text((vx - 22, y), "‹", font=font, fill=PAL["magenta"])
            d.text((vx, y), val, font=font, fill=col)
            w = d.textlength(val, font=font)
            d.text((vx + w + 8, y), "›", font=font, fill=PAL["magenta"])
        else:
            d.text((vx, y), val, font=font, fill=col)
        y += lh

    # footer
    d.line([pad, H - 44, W - pad, H - 44], fill=PAL["muted"], width=1)
    x = pad
    if fps is not None:
        fcol = PAL["lime"] if fps >= 55 else PAL["yellow"] if fps >= 30 else PAL["error"]
        d.text((x, H - 34), "FPS", font=font_s, fill=PAL["fg"])
        d.text((x + 42, H - 34), f"{fps:.0f}", font=font_s, fill=fcol)
        x += 90
    if status is not None:
        label, value = status
        d.text((x, H - 34), label, font=font_s, fill=PAL["fg"])
        d.text((x + d.textlength(label, font=font_s) + 12, H - 34), value, font=font_s,
               fill=PAL["cyan"])
    hint = "wheel change · click set"
    d.text((W - pad - d.textlength(hint, font=font_s), H - 34), hint, font=font_s,
           fill=PAL["muted"])
    return img, hits


class Panel:
    def __init__(self, ctx):
        self.ctx = ctx
        self.prog = ctx.program(vertex_shader=PANEL_VS, fragment_shader=PANEL_FS)
        self.prog["tex"] = 0
        unit = np.array([0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1], dtype="f4")
        self.vao = ctx.vertex_array(self.prog, [(ctx.buffer(unit.tobytes()), "2f", "in_uv")])
        self.tex = ctx.texture((W, H), 4)
        self.tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.tex.repeat_x = False         # clamp: CA/glitch sample past the edge
        self.tex.repeat_y = False
        self._sig = None
        self.hits = []

    def update(self, tab_names, tab_idx, rows, fps=None, status=None) -> None:
        """Repaint the bitmap, only when something shown has changed."""
        sig = (tab_idx, tuple(tab_names),
               tuple((r.get("section", False), r.get("label"), r.get("value"),
                      r.get("selected")) for r in rows),
               None if fps is None else int(fps), status)
        if sig == self._sig:
            return
        self._sig = sig
        img, self.hits = paint(tab_names, tab_idx, rows, fps, status)
        self.tex.write(np.asarray(img).tobytes())

    def draw(self, rect, t=0.0, appear=1.0, off=0.0) -> None:
        """Draw over `rect` (NDC) in the current viewport, shifted by `off` in x."""
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.prog["rect"].value = rect
        self.prog["off"].value = (off, 0.0)
        self.prog["time"].value = float(t)
        self.prog["appear"].value = float(appear)
        self.tex.use(0)
        self.vao.render(moderngl.TRIANGLES)
        self.ctx.disable(moderngl.BLEND)
