"""app.py — live desktop -> stereo 3D, orchestration.

Display loop runs at vsync (60fps). In DEPTH mode the depth model runs on a
background thread; the display reuses the most-recent depth map, so motion stays
smooth even though depth refreshes slower (async decoupling). FLAT mode skips
depth entirely (2D, both eyes identical).
"""
import argparse
import json
import os
import threading
import time

# project-local config (under braindance/config/, not ~/.config)
_STATE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config", "state.json"))


def load_state():
    try:
        with open(_STATE) as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(d):
    try:
        os.makedirs(os.path.dirname(_STATE), exist_ok=True)
        with open(_STATE, "w") as f:
            json.dump(d, f)
    except Exception:
        pass


def _migrate_tear_fill(v):
    """tear_fill was briefly a bool (True = bg-stretch). Map legacy bools onto the
    ramp/stretch/inpaint enum; pass valid strings through, default to inpaint."""
    if isinstance(v, bool):
        return "stretch" if v else "ramp"
    return v if v in ("ramp", "stretch", "inpaint") else "inpaint"


# every panel-adjustable setting is persisted (and reloaded on next launch)
_PERSIST = ("mode", "model", "trt", "sync", "ema", "disp", "screen_depth",
            "climit", "zoom", "ss",
            "tear", "tear_thresh", "tear_fill", "sharp_mode",
            "jbu_color", "jbu_reach", "jbu_depth", "gf_eps", "gf_radius")


def save_settings(S):
    save_state({k: getattr(S, k) for k in _PERSIST})


def _persist_sig(S):
    return (S.mode, S.model, S.trt, S.sync, round(S.ema, 2), round(S.disp, 2),
            round(S.screen_depth, 1), round(S.climit, 1),
            round(S.zoom, 3), S.ss, S.tear, round(S.tear_thresh, 3), S.tear_fill,
            S.sharp_mode, round(S.jbu_color, 2), round(S.jbu_reach, 2),
            round(S.jbu_depth, 2), round(S.gf_eps, 4), int(S.gf_radius))


# --- per-stage frame profiler (set BRAINDANCE_PROF=1) ---------------------
# Prints the wall-clock of each main-loop stage every 120 frames. In prof mode a
# glFinish runs after present so `gpu_finish` = real GPU present time and `swap` =
# vsync idle (large swap => GPU has headroom, you're vsync-bound; large gpu_finish
# with ~0 swap => compute-bound).
_PROF_KEYS = ("input", "capture", "composite", "depth", "present", "gpu_finish", "swap")


def _prof(acc, key, t):
    n = time.perf_counter()
    acc[key] += n - t
    return n


def _dump_prof(acc):
    n = max(1, acc["_n"])
    line = "  ".join(f"{k}={acc[k] / n * 1000:4.1f}" for k in _PROF_KEYS)
    tot = sum(acc[k] for k in _PROF_KEYS) / n * 1000
    print(f"[prof] {line} | frame={tot:5.1f}ms {1000.0 / max(tot, 1e-3):.0f}fps", flush=True)
    for k in _PROF_KEYS:
        acc[k] = 0.0
    acc["_n"] = 0

from types import SimpleNamespace

from .overlay import Overlay
from .capture import DesktopCapture
from .render import Renderer
from .input import Controls
from .depth import build_depth, registry
from .menu import Setting, Section, Tab, Menu


class DepthWorker(threading.Thread):
    """Runs the depth model on the latest submitted frame (drops stale frames)."""
    def __init__(self, depth):
        super().__init__(daemon=True)
        self.depth = depth
        self._frame = None
        self._result = None
        self._busy = False
        self._lock = threading.Lock()
        self._ev = threading.Event()
        self._stop = False

    def want_frame(self):
        # only feed it when idle, so we don't readback at 60fps for a ~30fps model
        with self._lock:
            return (not self._busy) and (self._frame is None)

    def submit(self, frame):
        with self._lock:
            self._frame = frame
        self._ev.set()

    def get(self):
        with self._lock:
            r, self._result = self._result, None
            return r

    def run(self):
        while not self._stop:
            self._ev.wait(0.1)
            self._ev.clear()
            with self._lock:
                f, self._frame = self._frame, None
                if f is not None:
                    self._busy = True
            if f is None:
                continue
            try:
                d = self.depth(f)
            except Exception as e:
                print("[braindance] depth error:", e, flush=True)
                d = None
            with self._lock:
                if d is not None:
                    self._result = d
                self._busy = False

    def stop(self):
        self._stop = True
        self._ev.set()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="depth-anything/Depth-Anything-V2-Small-hf")
    ap.add_argument("--in-h", type=int, default=392, help="depth model input height")
    ap.add_argument("--depth-h", type=int, default=700,
                    help="readback height for depth (>= the model's in_h so hi-res "
                         "models get real detail; low-res models just downsize it)")
    ap.add_argument("--disp", type=float, default=18.0, help="max disparity px (3D strength)")
    ap.add_argument("--screen-depth", type=float, default=0.0,
                    help="plane slide px: floats the whole scene toward/away (+ = closer)")
    ap.add_argument("--grid", type=int, default=1024,
                    help="warp mesh density. 2048 oversamples the depth map for ~3x the "
                         "present cost; 1024 matches it with no visible loss (no tear).")
    ap.add_argument("--ss", type=float, default=1.0,
                    help="present supersampling factor (e.g. 2.0) — antialiases the "
                         "depth-edge ladder. 1.0 = off. Costs ~ss^2 present fragments.")
    ap.add_argument("--mode", choices=["flat", "depth"], default="flat")
    ap.add_argument("--trt", action="store_true", help="use TensorRT FP16 depth engine (fast path)")
    ap.add_argument("--trt-model", default="da2-small",
                    help="registry model name for --trt (see models/registry.py)")
    ap.add_argument("--panel", default=None, help="WxH override, e.g. 3840x1080")
    ap.add_argument("--overlay-rect", default=None,
                    help="glasses output rect 'X,Y,W,H' to place + size the overlay "
                         "(default: primary monitor at 0,0)")
    ap.add_argument("--region", default=None,
                    help="desktop capture rect 'X,Y,W,H' in root px. Default: the "
                         "panel's left half. Point this at a separate output (e.g. "
                         "the eDP head) to grab a real desktop with no stacking race.")
    ap.add_argument("--sync", action="store_true", help="compute depth inline (debug)")
    ap.add_argument("--no-interop", action="store_true",
                    help="disable GL<->CUDA interop (the zero-copy sync depth path)")
    args = ap.parse_args(argv)

    panel = tuple(int(v) for v in args.panel.split("x")) if args.panel else None
    orect = tuple(int(v) for v in args.overlay_rect.split(",")) if args.overlay_rect else None

    ov = Overlay(panel=panel, rect=orect)
    if args.region:
        rx, ry, rw, rh = (int(v) for v in args.region.split(","))
    else:
        rx, ry, rw, rh = 0, 0, ov.eye_w, ov.ph   # legacy: left half of the panel
    region = (rx, ry, rw, rh)
    print(f"[braindance] panel {ov.pw}x{ov.ph}  eye {ov.eye_w}x{ov.ph}  region {region}", flush=True)

    cap = DesktopCapture(ov.ctx, ov.x_display, ov.xid, region)
    renderer = Renderer(ov.ctx, ov.pw, ov.ph, ov.eye_w, (rw, rh),
                        grid=args.grid, depth_h=args.depth_h, ss=args.ss)
    print("[braindance] loading depth model...", flush=True)
    saved = load_state()
    # all live-adjustable state in one place so the settings menu can drive it;
    # every field is loaded from the persisted state (falling back to CLI args)
    S = SimpleNamespace(
        mode=saved.get("mode", args.mode),
        model=saved.get("model", args.trt_model),    # registry key
        trt=bool(saved.get("trt", args.trt)),
        sync=bool(saved.get("sync", args.sync)),     # inline depth vs async worker
        ema=float(saved.get("ema", 0.5)),            # temporal depth smoothing 0..0.95
        disp=float(saved.get("disp", args.disp)),
        screen_depth=float(saved.get("screen_depth", args.screen_depth)),  # plane slide px
        climit=float(saved.get("climit", 0.0)),      # comfort disparity clamp px (0=off)
        zoom=float(saved.get("zoom", 1.0)),
        ss=max(1.0, float(saved.get("ss", args.ss))),
        tear=bool(saved.get("tear", False)),         # mesh tearing (crisp pop-out)
        tear_thresh=float(saved.get("tear_thresh", 0.08)),
        # disocclusion fill: ramp (legacy) / stretch (bg edge) / inpaint (continuation)
        tear_fill=_migrate_tear_fill(saved.get("tear_fill", "inpaint")),
        # edge-aware depth refine: off / jbu (joint-bilateral) / guided (guided filter)
        sharp_mode=saved.get("sharp_mode", "jbu" if saved.get("sharpen") else "off"),
        jbu_color=float(saved.get("jbu_color", saved.get("sharp_amt", 0.5))),  # color sigma
        jbu_reach=float(saved.get("jbu_reach", 1.5)),  # gather-stride multiplier
        jbu_depth=float(saved.get("jbu_depth", 0.0)),  # depth-range sigma (0 = off)
        gf_eps=float(saved.get("gf_eps", 0.01)),       # guided-filter regularization
        gf_radius=int(saved.get("gf_radius", 4)),      # guided-filter window radius
        tear_debug=False,                            # transient — paints torn gaps
        depth_view="off",                            # transient — off / 2d / 3d
    )
    panx, pany = 0.0, 0.0
    panel_open = False
    panel_open_t = 0.0          # wall time the panel last opened (for the materialize)

    try:
        depth = build_depth(S.model, S.trt, args.in_h)
    except Exception as e:                            # e.g. persisted TRT, no engine
        print(f"[braindance] {e}\n[braindance] falling back to torch", flush=True)
        S.trt = False
        depth = build_depth(S.model, S.trt, args.in_h)
    applied_backend = (S.trt, S.model)
    controls = Controls()

    worker = None
    if not S.sync:
        worker = DepthWorker(depth)
        worker.start()

    # GL<->CUDA interop: zero-copy sync depth path (no GPU->CPU->GPU round-trip, no
    # readback stall). Only valid on the GL/main thread => sync mode, and only for a
    # backend with a GPU-in/GPU-out path (TRT). Falls back silently if unavailable.
    interop_enabled = not args.no_interop
    interop_active = False

    def setup_interop():
        nonlocal interop_active
        if not (interop_enabled and S.sync and hasattr(depth, "infer_gpu")):
            return
        try:
            renderer.setup_interop(*depth.out_size)
            interop_active = True
            print("[braindance] GL<->CUDA interop ON (zero-copy depth)", flush=True)
        except Exception as e:
            interop_active = False
            print(f"[braindance] interop unavailable: {e}", flush=True)

    def teardown_interop():
        nonlocal interop_active
        if interop_active:
            renderer.teardown_interop()
            interop_active = False

    setup_interop()

    def swap_depth():
        """Rebuild the depth backend live (TENSORRT toggle / MODEL change)."""
        nonlocal depth, worker, applied_backend
        renderer.set_hud("loading %s%s..." % (S.model, " TRT" if S.trt else ""))
        try:
            new = build_depth(S.model, S.trt, args.in_h)
        except Exception as e:
            print(f"[braindance] backend swap failed: {e}", flush=True)
            S.trt, S.model = applied_backend     # revert the request
            return
        if worker is not None:
            worker.stop()
            worker = DepthWorker(new)
            worker.start()
        depth = new
        applied_backend = (S.trt, S.model)
        teardown_interop()      # re-wire interop to the new backend's output size
        setup_interop()
        print(f"[braindance] backend -> {depth.name}", flush=True)

    model_opts = list(registry.MODELS)
    is_jbu = lambda: S.sharp_mode == "jbu"
    is_gf = lambda: S.sharp_mode == "guided"
    menu = Menu([
        Tab("DISPLAY", [
            Setting("MODE", "enum", lambda: S.mode, lambda v: setattr(S, "mode", v),
                    options=["flat", "depth"], fmt=lambda v: v.upper(),
                    help="FLAT = 2D (both eyes identical). DEPTH = per-eye mesh-warp 3D."),
            Setting("ZOOM", "num", lambda: S.zoom, lambda v: setattr(S, "zoom", v),
                    step=0.1, lo=0.2, hi=8.0, fmt=lambda v: f"{v:.2f}x",
                    help="Scale the scene. Also bound to Super+Alt+scroll."),
            Setting("SUPERSAMPLE", "enum", lambda: S.ss, lambda v: setattr(S, "ss", v),
                    options=[1.0, 1.5, 2.0, 3.0], fmt=lambda v: f"{v:g}x",
                    help="Render the warp at NxN then box-downsample. Antialiases the "
                         "depth-edge ladder; costs ~N^2 fragments."),
        ]),
        Tab("DEPTH", [
            Setting("MODEL", "enum", lambda: S.model, lambda v: setattr(S, "model", v),
                    options=model_opts,
                    help="Monocular depth network (see models/registry.py). -hi variants "
                         "run at higher input res: sharper edges, slower."),
            Setting("TENSORRT", "bool", lambda: S.trt, lambda v: setattr(S, "trt", v),
                    help="Use the prebuilt TensorRT FP16 engine (~2x faster than torch). "
                         "Falls back to torch if no engine is built."),
            Setting("SYNC", "bool", lambda: S.sync, lambda v: setattr(S, "sync", v),
                    help="ON = compute depth inline (matched, no 1-frame lag, slower). "
                         "OFF = async worker thread, display reuses latest map."),
            Setting("DEPTH EMA", "num", lambda: S.ema, lambda v: setattr(S, "ema", v),
                    step=0.05, lo=0.0, hi=0.95, fmt=lambda v: f"{v:.2f}",
                    help="Temporal smoothing of the depth map (exponential moving "
                         "average). Higher = steadier depth, less flicker/boiling, but "
                         "more ghosting on motion. 0 = off."),
        ]),
        Tab("3D", [
            Section("GEOMETRY"),
            Setting("3D STRENGTH", "num", lambda: S.disp, lambda v: setattr(S, "disp", v),
                    step=1, lo=0, hi=60, fmt=lambda v: f"{v:.0f} px",
                    help="Disparity scale — how much total depth separation (parallax). "
                         "Super+Alt + = / -."),
            Setting("SCREEN DEPTH", "num", lambda: S.screen_depth,
                    lambda v: setattr(S, "screen_depth", v),
                    step=1, lo=-40, hi=40, fmt=lambda v: f"{v:+.0f} px",
                    help="Float the whole scene toward/away from you (constant per-eye "
                         "shift). One value works in FLAT and DEPTH: sets the comfortable "
                         "plane distance / IPD trim. + = closer, - = farther. "
                         "Super+Alt + [ / ]."),
            Setting("COMFORT", "num", lambda: S.climit, lambda v: setattr(S, "climit", v),
                    step=1, lo=0, hi=60, fmt=lambda v: "OFF" if v < 0.5 else f"{v:.0f} px",
                    help="Clamp on max |disparity| (px). Caps how far things pop out or "
                         "diverge, to avoid eye strain. 0 = off."),
        ]),
        Tab("TEAR", [
            Setting("DEPTH TEAR", "bool", lambda: S.tear, lambda v: setattr(S, "tear", v),
                    help="Drop mesh triangles spanning a depth cliff so foreground "
                         "silhouettes pop crisply instead of rubber-sheeting."),
            Setting("TEAR FILL", "enum", lambda: S.tear_fill,
                    lambda v: setattr(S, "tear_fill", v),
                    options=["ramp", "stretch", "inpaint"], fmt=lambda v: v.upper(),
                    help="Disocclusion gap behind a torn silhouette. RAMP = legacy "
                         "fg->bg smear (no-op). STRETCH = nearest bg edge pixel. "
                         "INPAINT = continue the background texture. Needs DEPTH TEAR; "
                         "most visible with SHARP MODE != off."),
            Setting("TEAR THRESH", "num", lambda: S.tear_thresh,
                    lambda v: setattr(S, "tear_thresh", v),
                    step=0.01, lo=0.02, hi=0.50, fmt=lambda v: f"{v:.2f}",
                    help="Normalized depth jump above which a triangle tears. Lower = "
                         "tears more readily."),
            Setting("TEAR DEBUG", "bool", lambda: S.tear_debug,
                    lambda v: setattr(S, "tear_debug", v),
                    help="Paint torn gaps bright magenta to see where tearing happens."),
        ]),
        Tab("REFINE", [
            Setting("SHARP MODE", "enum", lambda: S.sharp_mode,
                    lambda v: setattr(S, "sharp_mode", v),
                    options=["off", "jbu", "guided"], fmt=lambda v: v.upper(),
                    help="Edge-aware depth upsample. OFF = raw low-res depth. JBU = "
                         "joint-bilateral. GUIDED = guided filter (fewer halos, "
                         "robust to texture; ~2-4x faster at the default model)."),
            Setting("JBU COLOR", "num", lambda: S.jbu_color,
                    lambda v: setattr(S, "jbu_color", v),
                    step=0.1, lo=0.0, hi=1.0, fmt=lambda v: f"{v:.1f}",
                    visible=is_jbu,
                    help="Color sigma. Higher = depth snaps tighter to color edges "
                         "(crisper, but texture can leak into depth)."),
            Setting("JBU REACH", "num", lambda: S.jbu_reach,
                    lambda v: setattr(S, "jbu_reach", v),
                    step=0.5, lo=1.0, hi=4.0, fmt=lambda v: f"{v:.1f}x",
                    visible=is_jbu,
                    help="Gather-stride multiplier. Widens how far the kernel searches "
                         "so it can pull a depth edge onto a misaligned color edge."),
            Setting("JBU DEPTH", "num", lambda: S.jbu_depth,
                    lambda v: setattr(S, "jbu_depth", v),
                    step=0.1, lo=0.0, hi=1.0, fmt=lambda v: f"{v:.1f}",
                    visible=is_jbu,
                    help="Depth-range sigma (0 = off). Rejects neighbors whose depth "
                         "differs a lot — curbs texture-copy on flat surfaces."),
            Setting("GF EPS", "num", lambda: S.gf_eps, lambda v: setattr(S, "gf_eps", v),
                    step=0.002, lo=0.0, hi=0.08, fmt=lambda v: f"{v:.3f}",
                    visible=is_gf,
                    help="Regularization. Lower = sharper edge-snapping; higher = more "
                         "smoothing in flat regions (less texture-copy)."),
            Setting("GF RADIUS", "num", lambda: S.gf_radius,
                    lambda v: setattr(S, "gf_radius", v),
                    step=1, lo=1, hi=16, fmt=lambda v: f"{int(v)}",
                    visible=is_gf,
                    help="Box-filter window (depth-res texels). Larger = smoother model "
                         "but cost grows with radius (~3ms at r=16 on hi-res depth)."),
        ]),
        Tab("DEBUG", [
            Setting("DEPTH VIEW", "enum", lambda: S.depth_view,
                    lambda v: setattr(S, "depth_view", v),
                    options=["off", "2d", "3d"], fmt=lambda v: v.upper(),
                    help="OFF = normal. 2D = depth map as grayscale (flat). 3D = depth "
                         "as grayscale, still warped (parallax preview)."),
        ]),
    ])

    fps_t, fps_n, fps, tick = time.time(), 0, 0.0, 0
    anim_t0 = time.time()       # small monotonic-ish base for shader animation time
    PROF = bool(os.environ.get("BRAINDANCE_PROF"))
    prof = {k: 0.0 for k in _PROF_KEYS}
    prof["_n"] = 0
    saved_sig = _persist_sig(S)
    try:
        while not ov.should_close():
            now = time.time()
            dt = max(1e-3, now - getattr(main, "_last", now))
            main._last = now
            tick += 1
            if PROF:
                _t = time.perf_counter()
            ov.poll()

            ev = controls.update(dt)
            if ev.quit:
                break
            # global chords (work whether or not the panel is open)
            S.zoom = min(8.0, max(0.2, S.zoom * ev.zoom_mult))
            panx += ev.pan_dx
            pany += ev.pan_dy
            S.disp = max(0.0, S.disp + ev.disp_delta)        # +/- 1 px per press
            if ev.screen_delta:
                S.screen_depth = min(40.0, max(-40.0, S.screen_depth + ev.screen_delta))
            if ev.reset:
                S.zoom, panx, pany = 1.0, 0.0, 0.0
            if ev.toggle:
                S.mode = "depth" if S.mode == "flat" else "flat"
            changed = bool(ev.disp_delta or ev.screen_delta)
            if ev.panel:
                panel_open = not panel_open
                if panel_open:
                    panel_open_t = now      # restart the materialize animation
                (controls.grab_menu_keys if panel_open else controls.ungrab_menu_keys)()

            # menu navigation (only while the panel is open; keys are grabbed)
            if panel_open:
                if ev.tab_next:
                    menu.next_tab()
                if ev.tab_prev:
                    menu.prev_tab()
                if ev.nav_up:
                    menu.up()
                if ev.nav_down:
                    menu.down()
                if ev.nav_left:
                    changed |= menu.left()
                if ev.nav_right:
                    changed |= menu.right()
                if ev.enter:
                    panel_open = False
                    controls.ungrab_menu_keys()

            # apply live changes that need GL/model rebuilds
            if S.ss != renderer.ss:
                renderer.set_ss(S.ss)
            if (S.trt, S.model) != applied_backend:
                swap_depth()
            # SYNC toggle: async depth worker (off) <-> inline depth (on)
            if S.sync and worker is not None:
                worker.stop()
                worker = None
                setup_interop()         # back on the GL thread -> zero-copy depth
            elif not S.sync and worker is None:
                teardown_interop()      # worker thread has no GL context
                worker = DepthWorker(depth)
                worker.start()
            # persist ANY setting change (mode/model/trt/sync/disp/screen_depth/zoom/ss)
            sig = _persist_sig(S)
            if sig != saved_sig:
                saved_sig = sig
                save_settings(S)

            if PROF:
                _t = _prof(prof, "input", _t)
            windows = cap.update()
            if PROF:
                _t = _prof(prof, "capture", _t)
            sig = tuple((w[1], w[2], w[3], w[4]) for w in windows)   # x,y,w,h per window
            if sig != getattr(main, "_winsig", None):
                main._winsig = sig
                # A restack (fullscreen toggle, new/closed window) can lift a real
                # window above our override-redirect overlay; since Composite is in
                # Automatic mode the server then paints that raw window on top,
                # occluding the warped scene + cursor. Reclaim top over the next few
                # frames — the WM's restack and resize don't land in one frame.
                main._raise_until = tick + 10
                print(f"[braindance] windows changed ({len(windows)}): "
                      + ", ".join(f"{w}x{h}@{x},{y}" for (_, x, y, w, h) in windows), flush=True)
            if tick < getattr(main, "_raise_until", 0) or tick % 60 == 1:
                ov.raise_top()         # event-driven; periodic % 60 is the backstop
            renderer.composite(windows)
            if PROF:
                _t = _prof(prof, "composite", _t)

            depth.ema_alpha = 1.0 - S.ema     # live temporal-smoothing strength
            if S.mode == "depth" or S.depth_view != "off":   # depth-view needs a map
                if worker is not None:
                    if worker.want_frame():                  # readback only when idle
                        f = renderer.depth_frame()
                        if f is not None:
                            worker.submit(f)
                    d = worker.get()
                    if d is not None:
                        renderer.set_depth(d)
                elif interop_active:
                    # sync, zero-copy: blit -> CUDA reads it -> infer -> CUDA writes
                    # the depth texture. No GPU->CPU->GPU round-trip, no readback stall.
                    renderer.write_depth_interop(depth.infer_gpu(renderer.depth_src_interop()))
                else:
                    # sync: read the current frame (matched depth, no 1-frame desync)
                    renderer.set_depth(depth(renderer.depth_frame(use_pbo=False)))

            if PROF:
                _t = _prof(prof, "depth", _t)
            # 3D depth view warps with disparity too, so it shows parallax even in FLAT
            disp = S.disp if (S.mode == "depth" or S.depth_view == "3d") else 0.0
            renderer.draw(disp, S.zoom, panx, pany,
                          pointer=cap.pointer(), tear=S.tear, tear_thresh=S.tear_thresh,
                          tear_fill=S.tear_fill,
                          sharp_mode=S.sharp_mode, jbu_color=S.jbu_color,
                          jbu_reach=S.jbu_reach, jbu_depth=S.jbu_depth,
                          gf_eps=S.gf_eps, gf_radius=int(S.gf_radius),
                          climit_px=S.climit, screen_depth_px=S.screen_depth,
                          tear_debug=S.tear_debug, depth_view=S.depth_view)

            if panel_open:
                names, ti = menu.tab_bar()
                renderer.set_panel(names, ti, menu.rows(), fps)
                appear = min(1.0, (now - panel_open_t) / 0.30)   # 0.3s materialize
                renderer.draw_panel(now - anim_t0, appear, screen_depth_px=S.screen_depth)

            if PROF:
                _t = _prof(prof, "present", _t)
                ov.ctx.finish()          # force GPU done so swap below = vsync idle
                _t = _prof(prof, "gpu_finish", _t)

            fps_n += 1
            if now - fps_t >= 1.0:                 # fps tracked for the OSD panel
                fps = fps_n / (now - fps_t)
                fps_t, fps_n = now, 0
            ov.swap()
            if PROF:
                _prof(prof, "swap", _t)
                prof["_n"] += 1
                if prof["_n"] >= 120:
                    _dump_prof(prof)
    finally:
        save_settings(S)
        controls.ungrab_menu_keys()
        if worker is not None:
            worker.stop()
        teardown_interop()
        controls.shutdown()
        cap.shutdown()
        ov.terminate()


if __name__ == "__main__":
    main()
