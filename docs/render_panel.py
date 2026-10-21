"""Render braindance's in-glasses settings panel offscreen, one PNG per tab.

Uses the real menu (settings.build_menu) over the default settings, and the
real panel bitmap (panel.paint), so no window, GPU context or glasses are
needed. Each tab is shown where it appears: the live desktop's tabs in a 3D
mode with the mesh warp (so EDGES and REFINE show), SPACE in a 2D mode with
the desktop anchored, MODEL while a model is open.

    uv run docs/render_panel.py docs
"""
import os
import sys
from types import SimpleNamespace

from PIL import Image

from braindance import panel, settings
from braindance.glasses import modes
from braindance.menu import Section, Setting

OUT = sys.argv[1]


def glasses(mode):
    return [
        Section("DISPLAY MODE"),
        Setting("OUTPUT", "enum", lambda: mode.name, lambda v: None,
                options=[m.name for m in modes.MODES], fmt=str.upper),
        Section("SESSION"),
        Setting("END SESSION", "action", lambda: None, None, fmt=lambda v: "CLICK"),
    ]


def save(menu, tab, fps, status):
    menu.select_tab(menu.tab_bar()[0].index(tab))
    img, _ = panel.paint(*menu.tab_bar(), menu.rows(), fps=fps, status=status)
    # the panel is translucent over the scene: show it over the dark ground it floats on
    ground = Image.new("RGBA", img.size, (6, 4, 14, 255))
    Image.alpha_composite(ground, img).convert("RGB").save(
        os.path.join(OUT, f"panel-{tab.lower()}.png"))
    print(f"{tab}: {len(menu.rows())} rows")


S = SimpleNamespace(**settings.DEFAULTS, fill_debug=False, depth_view="off")
S.live3d, S.warp = True, "mesh"
desktop = settings.build_menu(S, glasses(modes.by_name("3d-1080p60")), lambda: True,
                              lambda: None, lambda: False)
for tab in desktop.tab_bar()[0]:
    save(desktop, tab, 60, ("DEPTH", "20/s"))

S2 = SimpleNamespace(**vars(S))
S2.space_mode = "anchor"
save(settings.build_menu(S2, glasses(modes.by_name("2d-1080p120")), lambda: False,
                         lambda: None, lambda: False), "SPACE", 120, ("TRACK", "6DOF"))
save(settings.build_menu(S, glasses(modes.by_name("3d-1200p90")), lambda: True,
                         lambda: None, lambda: True), "MODEL", 90, None)
