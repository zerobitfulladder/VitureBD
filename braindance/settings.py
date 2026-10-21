"""settings.py — the live-tunable settings, and the panel's menu over them.

Every setting persists to config/state.json the moment it changes, and comes
back on the next session. Two are transient (fill_debug, depth_view).
"""
import json
import os
from types import SimpleNamespace

from . import registry
from .menu import Menu, Section, Setting, Tab

STATE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "config", "state.json")

DEFAULTS = {
    "live3d": False,          # FLAT (both eyes the same) or LIVE 3D (per-eye warp)
    "model": "da2-small",     # registry key
    "trt": True,              # TensorRT engine (falls back to torch without one)
    "sync": False,            # depth inline vs on a worker thread
    "range_ema": 0.75,        # depth-range smoothing over frames (iw3's 0.75), 0 = per frame
    "stabilize": 0.5,         # smoothing of still pixels' depth, 0..0.95
    "warp": "mlbw",           # stereo warp: mlbw (learned, stereo.py) or mesh
    "dilation": 2,            # edge dilation passes (iw3's default is 2), 0 = off
    "disp": 18.0,             # 3D strength: max disparity, px
    "screen_depth": 0.0,      # plane slide, px (+ = closer)
    "climit": 0.0,            # comfort disparity clamp, px (0 = off)
    "zoom": 1.0,
    "ss": 1.0,                # present supersampling (mesh warp)
    "edge_fill": "network",   # mesh, the uncovered background: off / mirror / network
    "edge_cliff": 4.0,        # px uncovered from which a stretch is a depth edge
    "edge_halo": 6,           # px of background rim replaced beside a depth edge
    "sharp_mode": "off",      # edge-aware depth refine: off / jbu / guided
    "jbu_color": 0.5,
    "jbu_reach": 1.5,
    "jbu_depth": 0.0,
    "gf_eps": 0.01,
    "gf_radius": 4,
    "media_depth": 0.0,       # video eye alignment, px per eye pair (+ = nearer)
    "media_size": 1.0,        # video picture size, fraction of the eye
    "media_volume": 80,
    "panel_pos": [0.97, 0.96],  # the panel's place: panel.place(), bottom-right
    "space_mode": "off",      # 2D: off (X drives the glasses) / anchor / follow
    "space_dof": "3dof",      # 2D tracking: 3dof (IMU) / 6dof (cameras + VIO too)
    "space_size": 0.9,        # 2D screen size, 1 = one desktop px per display px
    "space_distance": 1.5,    # 2D screen distance, m: 6DOF's lean, CURVE's radius
    "space_curve": 1.0,       # 2D screen curve: 1 = around the eyes, 0 = flat
    "space_follow": 0.3,      # FOLLOW's catch-up time constant, s
    "space_predict": 12.0,    # head pose prediction, ms (the screen and models)
    "model_mode": "anchor",   # a 3D model: anchor (in the room) / follow
    "model_dof": "6dof",      # its tracking: 3dof / 6dof
    "model_size": 0.4,        # its largest side, m
    "model_distance": 1.0,    # how far in front it is placed, m
    "model_follow": 0.3,      # FOLLOW's catch-up time constant, s
    "model_exposure": 1.0,    # its lighting's brightness
}


def load() -> SimpleNamespace:
    try:
        with open(STATE) as f:
            saved = json.load(f)
    except (OSError, ValueError):
        saved = {}
    S = SimpleNamespace(**{k: saved.get(k, v) for k, v in DEFAULTS.items()})
    S.fill_debug = False      # paints the edge fill
    S.depth_view = "off"      # off / 2d / 3d
    return S


def save(S) -> None:
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    with open(STATE, "w") as f:
        json.dump(signature(S), f)


def signature(S) -> dict:
    return {k: getattr(S, k) for k in DEFAULTS}


def build_menu(S, glasses: list, in_3d, recenter, viewing) -> Menu:
    """The panel's menu. `glasses` is the GLASSES tab's items (the session owns
    the glasses); SPACE means something only in 2D, MODEL only while a 3D model
    is open (`viewing()`), every other tab only for the live desktop in 3D
    (`in_3d()`, no model open). `recenter()`
    places the 2D screen or the model in front of you again."""
    is_jbu = lambda: S.sharp_mode == "jbu"
    is_gf = lambda: S.sharp_mode == "guided"
    is_mesh = lambda: S.warp == "mesh"
    desktop_3d = lambda: in_3d() and not viewing()     # the live desktop's tabs
    mesh_3d = lambda: desktop_3d() and is_mesh()
    in_2d = lambda: not in_3d()
    return Menu([
        Tab("GLASSES", glasses),
        Tab("SPACE", [
            Section("PLACEMENT"),
            Setting("MODE", "enum", lambda: S.space_mode, lambda v: setattr(S, "space_mode", v),
                    options=["off", "anchor", "follow"], fmt=lambda v: v.upper(),
                    help="OFF = X drives the glasses directly: the sharpest picture, "
                         "the least delay. ANCHOR = the screen stays where you put it "
                         "while you look around. FOLLOW = it glides after your head, "
                         "trailing behind, and settles in front of you when you stop."),
            Setting("TRACKING", "enum", lambda: S.space_dof, lambda v: setattr(S, "space_dof", v),
                    options=["3dof", "6dof"], fmt=lambda v: v.upper(),
                    help="3DOF = head rotation (the IMU). 6DOF = position too, from "
                         "the glasses' cameras: lean in and the screen stays in the "
                         "room. 6DOF costs CPU; changing it restarts the glasses' SDK."),
            Setting("RECENTER", "action", lambda: None, None, fmt=lambda v: "CLICK",
                    on_click=recenter,
                    help="Place the screen in front of you again. So do the mouse's "
                         "back side button and a middle click beside the panel."),
            Section("SCREEN"),
            Setting("SIZE", "num", lambda: S.space_size, lambda v: setattr(S, "space_size", v),
                    step=0.05, lo=0.3, hi=2.0, fmt=lambda v: f"{v * 100:.0f}%",
                    help="100% = one desktop pixel per display pixel, the screen "
                         "filling the view. The wheel beside the panel sizes it too."),
            Setting("CURVE", "num", lambda: S.space_curve, lambda v: setattr(S, "space_curve", v),
                    step=0.1, lo=0.0, hi=1.0,
                    fmt=lambda v: "FLAT" if v < 0.05 else f"{v * 100:.0f}%",
                    help="Bend the screen around you like a curved monitor: 100% = "
                         "every column the same distance from your eyes. Only FLAT "
                         "keeps it pixel for pixel when it is straight ahead."),
            Setting("DISTANCE", "num", lambda: S.space_distance,
                    lambda v: setattr(S, "space_distance", v),
                    step=0.1, lo=0.5, hi=5.0, fmt=lambda v: f"{v:.1f} m",
                    visible=lambda: S.space_dof == "6dof",
                    help="How far away the screen is placed: how much it grows as "
                         "you lean in, and the curve's radius."),
            Setting("FOLLOW", "num", lambda: S.space_follow,
                    lambda v: setattr(S, "space_follow", v),
                    step=0.05, lo=0.05, hi=1.5, fmt=lambda v: f"{v:.2f} s",
                    visible=lambda: S.space_mode == "follow",
                    help="How far the screen trails your head: the time it takes "
                         "to catch up most of the way."),
            Setting("PREDICT", "num", lambda: S.space_predict,
                    lambda v: setattr(S, "space_predict", v),
                    step=1, lo=0, hi=40, fmt=lambda v: f"{v:.0f} ms",
                    help="How far ahead the head pose is predicted, to cover the time "
                         "until the frame is lit. Too little and the screen trails "
                         "your turns; too much and it swings past."),
        ], visible=in_2d),
        Tab("MODEL", [
            Section("PLACEMENT"),
            Setting("MODE", "enum", lambda: S.model_mode, lambda v: setattr(S, "model_mode", v),
                    options=["anchor", "follow"], fmt=lambda v: v.upper(),
                    help="ANCHOR = the model stays where it was placed, in the room. "
                         "FOLLOW = it glides after your head and settles in front of "
                         "you when you stop."),
            Setting("TRACKING", "enum", lambda: S.model_dof, lambda v: setattr(S, "model_dof", v),
                    options=["3dof", "6dof"], fmt=lambda v: v.upper(),
                    help="6DOF = walk around it and lean in. 3DOF = head rotation only. "
                         "Changing it restarts the glasses' SDK."),
            Setting("RECENTER", "action", lambda: None, None, fmt=lambda v: "CLICK",
                    on_click=recenter,
                    help="Place the model in front of you again. So do the mouse's "
                         "back side button, a middle click and R."),
            Section("MODEL"),
            Setting("SIZE", "num", lambda: S.model_size, lambda v: setattr(S, "model_size", v),
                    step=0.05, lo=0.05, hi=5.0, fmt=lambda v: f"{v * 100:.0f} cm",
                    help="Its largest side. The wheel and - / = size it too."),
            Setting("DISTANCE", "num", lambda: S.model_distance,
                    lambda v: setattr(S, "model_distance", v),
                    step=0.1, lo=0.3, hi=5.0, fmt=lambda v: f"{v:.1f} m",
                    help="How far in front of you it is placed."),
            Setting("FOLLOW", "num", lambda: S.model_follow,
                    lambda v: setattr(S, "model_follow", v),
                    step=0.05, lo=0.05, hi=1.5, fmt=lambda v: f"{v:.2f} s",
                    visible=lambda: S.model_mode == "follow",
                    help="How far the model trails your head: the time it takes to "
                         "catch up most of the way."),
            Setting("EXPOSURE", "num", lambda: S.model_exposure,
                    lambda v: setattr(S, "model_exposure", v),
                    step=0.1, lo=0.2, hi=3.0, fmt=lambda v: f"{v:.1f}x",
                    help="How bright the model's lighting is."),
            Setting("PREDICT", "num", lambda: S.space_predict,
                    lambda v: setattr(S, "space_predict", v),
                    step=1, lo=0, hi=40, fmt=lambda v: f"{v:.0f} ms",
                    help="How far ahead the head pose is predicted (shared with the "
                         "2D screen's): raise it if the model trails your turns, lower "
                         "it if it swings past."),
        ], visible=viewing),
        Tab("DISPLAY", [
            Setting("LIVE 3D", "bool", lambda: S.live3d, lambda v: setattr(S, "live3d", v),
                    help="OFF = both eyes see the same picture. ON = each eye is warped by "
                         "the live depth map."),
            Setting("ZOOM", "num", lambda: S.zoom, lambda v: setattr(S, "zoom", v),
                    step=0.1, lo=0.2, hi=8.0, fmt=lambda v: f"{v:.2f}x",
                    help="Scale the scene. Also the wheel beside the panel."),
            Setting("SUPERSAMPLE", "enum", lambda: S.ss, lambda v: setattr(S, "ss", v),
                    options=[1.0, 1.5, 2.0, 3.0], fmt=lambda v: f"{v:g}x", visible=is_mesh,
                    help="Render the mesh warp at NxN then box-downsample. Antialiases "
                         "the depth-edge ladder; costs ~N^2 fragments."),
        ], visible=desktop_3d),
        Tab("DEPTH", [
            Setting("MODEL", "enum", lambda: S.model, lambda v: setattr(S, "model", v),
                    options=list(registry.MODELS),
                    help="Monocular depth network (see braindance/registry.py). -hi "
                         "variants run at higher input res: sharper edges, slower."),
            Setting("TENSORRT", "bool", lambda: S.trt, lambda v: setattr(S, "trt", v),
                    help="Use the prebuilt TensorRT FP16 engine (~2x faster than torch). "
                         "Falls back to torch if no engine is built."),
            Setting("SYNC", "bool", lambda: S.sync, lambda v: setattr(S, "sync", v),
                    help="ON = depth inside each frame: it matches the frame exactly, but "
                         "the frame waits for it (30 fps or less on a laptop GPU). OFF = "
                         "depth on a worker thread: the display holds 60 fps and reuses "
                         "the latest depth."),
            Section("STABILITY"),
            Setting("RANGE SMOOTH", "num", lambda: S.range_ema,
                    lambda v: setattr(S, "range_ema", v),
                    step=0.05, lo=0.0, hi=0.95, fmt=lambda v: "OFF" if v < 0.01 else f"{v:.2f}",
                    help="Smooths the depth range over frames, so the whole scene stops "
                         "breathing when something near comes into view. Resets at "
                         "cuts. iw3 uses 0.75. 0 = a fresh range every frame."),
            Setting("STABILIZE", "num", lambda: S.stabilize,
                    lambda v: setattr(S, "stabilize", v),
                    step=0.05, lo=0.0, hi=0.95, fmt=lambda v: "OFF" if v < 0.01 else f"{v:.2f}",
                    help="Holds still pixels' depth steady (less flicker). Pixels whose "
                         "colour changes take the new depth at once, so moving things "
                         "don't trail. Higher = steadier. 0 = off."),
        ], visible=desktop_3d),
        Tab("3D", [
            Section("WARP"),
            Setting("WARP", "enum", lambda: S.warp, lambda v: setattr(S, "warp", v),
                    options=["mlbw", "mesh"], fmt=lambda v: v.upper(),
                    help="MLBW = iw3's learned warp: blends two samples per pixel, no "
                         "holes, cleaner edges; needs its engines (braindance-build-"
                         "engine mlbw). MESH = the depth-displaced mesh, with the EDGES "
                         "and REFINE tabs."),
            Setting("EDGE DILATION", "num", lambda: S.dilation,
                    lambda v: setattr(S, "dilation", v),
                    step=1, lo=0, hi=4, fmt=lambda v: "OFF" if v == 0 else f"{int(v)}",
                    help="Pulls the depth edges of near things in by about a pixel per "
                         "pass, so the background beside them isn't dragged along. "
                         "iw3's default is 2. 0 = off."),
            Section("GEOMETRY"),
            Setting("3D STRENGTH", "num", lambda: S.disp, lambda v: setattr(S, "disp", v),
                    step=1, lo=0, hi=60, fmt=lambda v: f"{v:.0f} px",
                    help="Disparity scale — how much total depth separation (parallax)."),
            Setting("SCREEN DEPTH", "num", lambda: S.screen_depth,
                    lambda v: setattr(S, "screen_depth", v),
                    step=1, lo=-40, hi=40, fmt=lambda v: f"{v:+.0f} px",
                    help="Float the whole scene toward/away from you (constant per-eye "
                         "shift). Works with LIVE 3D on or off: sets the comfortable "
                         "plane distance / IPD trim. + = closer, - = farther."),
            Setting("COMFORT", "num", lambda: S.climit, lambda v: setattr(S, "climit", v),
                    step=1, lo=0, hi=60, fmt=lambda v: "OFF" if v < 0.5 else f"{v:.0f} px",
                    help="Clamp on max |disparity| (px). Caps how far things pop out or "
                         "diverge, to avoid eye strain. 0 = off."),
        ], visible=desktop_3d),
        Tab("EDGES", [
            Setting("EDGE FILL", "enum", lambda: S.edge_fill, lambda v: setattr(S, "edge_fill", v),
                    options=["off", "mirror", "network"], fmt=lambda v: v.upper(),
                    help="Where a near thing's edge moves off the background, the mesh "
                         "stretches the edge across the gap it leaves (streaks). MIRROR "
                         "fills the gap with the background beside it, mirrored (60 fps "
                         "with SYNC). NETWORK has iw3's inpainting network paint it: "
                         "about 15 ms more per frame with gaps, so 30 fps with SYNC. "
                         "Needs braindance-build-engine inpaint."),
            Setting("CLIFF", "num", lambda: S.edge_cliff, lambda v: setattr(S, "edge_cliff", v),
                    step=1, lo=2, hi=12, fmt=lambda v: f"{v:.0f} px",
                    help="How wide a gap (px) counts as a depth edge to fill. Lower "
                         "fills more; higher leaves smaller gaps stretched."),
            Setting("HALO", "num", lambda: S.edge_halo, lambda v: setattr(S, "edge_halo", v),
                    step=1, lo=0, hi=12, fmt=lambda v: f"{int(v)} px",
                    help="Background beside the gap also replaced (px). It carries the "
                         "near thing's blurred rim, which would outline the fill."),
            Setting("FILL DEBUG", "bool", lambda: S.fill_debug,
                    lambda v: setattr(S, "fill_debug", v),
                    help="Paint the fill magenta to see where it happens."),
        ], visible=mesh_3d),
        Tab("REFINE", [
            Setting("SHARP MODE", "enum", lambda: S.sharp_mode,
                    lambda v: setattr(S, "sharp_mode", v),
                    options=["off", "jbu", "guided"], fmt=lambda v: v.upper(),
                    help="Edge-aware depth upsample. OFF = raw low-res depth. JBU = "
                         "joint-bilateral. GUIDED = guided filter (fewer halos, "
                         "robust to texture; ~2-4x faster at the default model)."),
            Setting("JBU COLOR", "num", lambda: S.jbu_color,
                    lambda v: setattr(S, "jbu_color", v),
                    step=0.1, lo=0.0, hi=1.0, fmt=lambda v: f"{v:.1f}", visible=is_jbu,
                    help="Color sigma. Higher = depth snaps tighter to color edges "
                         "(crisper, but texture can leak into depth)."),
            Setting("JBU REACH", "num", lambda: S.jbu_reach,
                    lambda v: setattr(S, "jbu_reach", v),
                    step=0.5, lo=1.0, hi=4.0, fmt=lambda v: f"{v:.1f}x", visible=is_jbu,
                    help="Gather-stride multiplier. Widens how far the kernel searches "
                         "so it can pull a depth edge onto a misaligned color edge."),
            Setting("JBU DEPTH", "num", lambda: S.jbu_depth,
                    lambda v: setattr(S, "jbu_depth", v),
                    step=0.1, lo=0.0, hi=1.0, fmt=lambda v: f"{v:.1f}", visible=is_jbu,
                    help="Depth-range sigma (0 = off). Rejects neighbors whose depth "
                         "differs a lot — curbs texture-copy on flat surfaces."),
            Setting("GF EPS", "num", lambda: S.gf_eps, lambda v: setattr(S, "gf_eps", v),
                    step=0.002, lo=0.0, hi=0.08, fmt=lambda v: f"{v:.3f}", visible=is_gf,
                    help="Regularization. Lower = sharper edge-snapping; higher = more "
                         "smoothing in flat regions (less texture-copy)."),
            Setting("GF RADIUS", "num", lambda: S.gf_radius,
                    lambda v: setattr(S, "gf_radius", v),
                    step=1, lo=1, hi=16, fmt=lambda v: f"{int(v)}", visible=is_gf,
                    help="Box-filter window (depth-res texels). Larger = smoother model "
                         "but cost grows with radius (~3ms at r=16 on hi-res depth)."),
        ], visible=mesh_3d),
        Tab("DEBUG", [
            Setting("DEPTH VIEW", "enum", lambda: S.depth_view,
                    lambda v: setattr(S, "depth_view", v),
                    options=["off", "2d", "3d"], fmt=lambda v: v.upper(),
                    help="OFF = normal. 2D = depth map as grayscale (flat). 3D = depth "
                         "as grayscale, still warped (parallax preview)."),
        ], visible=desktop_3d),
    ])
