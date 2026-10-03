"""Render braindance's in-glasses settings panel offscreen, one PNG per tab.

Uses the real menu definition (the block in app.py, executed as written), the
saved settings in config/state.json, and Renderer.set_panel itself. Only the GL
texture upload is replaced by a capture of the PIL image, so no window, GPU
context or glasses are needed.

    cd wreath/braindance && .venv/bin/python docs/render_panel.py docs
"""
import json
import os
import sys
import textwrap
from types import SimpleNamespace

ROOT = os.path.expanduser("~/Projects/VitureBD/wreath/braindance")
OUT = sys.argv[1]
sys.path.insert(0, ROOT)

from PIL import Image, ImageFont  # noqa: E402

from braindance.depth import registry  # noqa: E402
from braindance.menu import Menu, Section, Setting, Tab  # noqa: E402
from braindance.render import Renderer  # noqa: E402

saved = json.load(open(os.path.join(ROOT, "config", "state.json")))
S = SimpleNamespace(
    mode=saved["mode"], model=saved["model"], trt=bool(saved["trt"]),
    sync=bool(saved["sync"]), ema=float(saved["ema"]), disp=float(saved["disp"]),
    screen_depth=float(saved["screen_depth"]), climit=float(saved["climit"]),
    zoom=float(saved["zoom"]), ss=float(saved["ss"]), tear=bool(saved["tear"]),
    tear_thresh=float(saved["tear_thresh"]), tear_fill=saved["tear_fill"],
    sharp_mode=saved["sharp_mode"], jbu_color=float(saved["jbu_color"]),
    jbu_reach=float(saved["jbu_reach"]), jbu_depth=float(saved["jbu_depth"]),
    gf_eps=float(saved["gf_eps"]), gf_radius=int(saved["gf_radius"]),
    tear_debug=False, depth_view="off",
)

# The menu block from app.py, executed as written.
src = open(os.path.join(ROOT, "braindance", "app.py")).read().splitlines()
start = next(i for i, l in enumerate(src) if l.strip().startswith("model_opts = "))
end = next(i for i in range(start, len(src)) if src[i] == "    ])")
ns = dict(S=S, registry=registry, Menu=Menu, Setting=Setting, Section=Section, Tab=Tab)
exec(textwrap.dedent("\n".join(src[start:end + 1])), ns)
menu = ns["menu"]


class Capture:
    def write(self, data):
        self.image = Image.frombytes("RGBA", (r.PANEL_W, r.PANEL_H), data)


r = Renderer.__new__(Renderer)
r.PANEL_W, r.PANEL_H = 640, 436
r.pfont = ImageFont.truetype("DejaVuSansMono.ttf", 21)
r.pfont_b = ImageFont.truetype("DejaVuSansMono-Bold.ttf", 26)
r.pfont_s = ImageFont.truetype("DejaVuSansMono.ttf", 16)
r._panel_sig = None
r.panel_tex = Capture()

names, _ = menu.tab_bar()
for i in range(len(names)):
    tab_names, ti = menu.tab_bar()
    r._panel_sig = None
    r.set_panel(tab_names, ti, menu.rows(), 60)
    # the panel is translucent over the scene: show it over the dark ground it floats on
    ground = Image.new("RGBA", r.panel_tex.image.size, (6, 4, 14, 255))
    Image.alpha_composite(ground, r.panel_tex.image).convert("RGB").save(
        os.path.join(OUT, f"panel-{ti}-{tab_names[ti].lower()}.png"))
    print("tab", ti, tab_names[ti], len(menu.rows()), "rows")
    menu.next_tab()
