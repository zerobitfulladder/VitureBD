"""desktop.py — the live desktop in a 3D (side-by-side) mode.

Captures the desktop window by window, estimates its depth, and draws a view
for each eye, one frame per vsync. With LIVE 3D off both eyes see the same
picture (SCREEN DEPTH still slides the plane nearer or farther); with it on,
each eye is warped by the depth: by the learned warp (stereo.py) or by the
depth-displaced mesh.

The depth pipeline (depth.py) is built the first time LIVE 3D needs it. By
default it runs on a worker thread and the display reuses its latest result,
so motion stays smooth while depth refreshes at its own rate. Either way its
results reach the renderer's textures straight from CUDA when the GL<->CUDA
interop is available.

The desktop draws into a presenter.Stage the session owns; while a video plays
on that stage the session simply stops calling frame().
"""
import os
import threading
import time

from .capture import DesktopCapture
from .depth import Pipeline, build_depth
from .render import Renderer

# Warp mesh density. It matches the depth map with no visible loss; 2048
# oversamples it for about 3x the present cost.
GRID = 1024
# Readback height for depth: at least the largest model's input height, so the
# high-resolution model gets real detail; low-resolution ones just downsize it.
DEPTH_H = 700


def log(*args) -> None:
    print("[braindance]", *args, flush=True)


class DepthWorker(threading.Thread):
    """Runs the pipeline on the latest submitted frame (drops stale frames)."""
    def __init__(self, pipeline):
        super().__init__(daemon=True)
        self.pipeline = pipeline
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
                r = self.pipeline.from_frame(f)
            except Exception as e:
                log("depth error:", e)
                r = None
            with self._lock:
                if r is not None:
                    self._result = r
                self._busy = False

    def stop(self):
        self._stop = True
        self._ev.set()


# The depth backend and the stereo engines outlive a Desktop: leaving 3D and
# coming back, or changing the 3D mode, reuses them.
_backend, _backend_key = None, None
_stereo = []            # [Stereo or None] once tried
_inpaint = []           # [Inpaint or None] once tried


def _get_backend(S):
    """The backend for S.model / S.trt. Without a TensorRT engine it falls back
    to torch and turns S.trt off."""
    global _backend, _backend_key
    if (S.trt, S.model) == _backend_key:
        return _backend
    log(f"loading depth model {S.model}{' (TensorRT)' if S.trt else ''}")
    try:
        backend = build_depth(S.model, S.trt)
    except Exception as e:
        if not S.trt:
            raise
        log(f"{e}\n[braindance] falling back to torch")
        S.trt = False
        backend = build_depth(S.model, False)
    _backend, _backend_key = backend, (S.trt, S.model)
    return backend


def _get_stereo():
    """The mlbw engines, or None (logged) if they aren't built."""
    if not _stereo:
        from .stereo import Stereo
        try:
            _stereo.append(Stereo())
        except FileNotFoundError as e:
            log(e)
            _stereo.append(None)
    return _stereo[0]


def _get_inpaint():
    """The inpainting engine for EDGE FILL = NETWORK, or None (logged) if it
    isn't built."""
    if not _inpaint:
        from .inpaint import Inpaint
        try:
            _inpaint.append(Inpaint())
        except FileNotFoundError as e:
            log(e)
            _inpaint.append(None)
    return _inpaint[0]


# --- per-stage frame profiler (set BRAINDANCE_PROF=1) ---------------------
# Prints the wall-clock of each frame stage every 120 frames. In prof mode a
# glFinish runs after present so `gpu_finish` = real GPU present time and `swap` =
# vsync idle (large swap => GPU has headroom, you're vsync-bound; large gpu_finish
# with ~0 swap => compute-bound).
_PROF_KEYS = ("capture", "composite", "depth", "present", "gpu_finish", "swap")


class Desktop:
    def __init__(self, stage, mode, S):
        """Take over `stage` (sized for `mode`, a side-by-side Mode) and capture the
        desktop into it."""
        self.S = S
        self.stage = stage
        self.eye_w, eh = mode.desktop
        self.cap = DesktopCapture(stage.ctx, stage.x_display, stage.xid, (0, 0, self.eye_w, eh))
        self.renderer = Renderer(stage.ctx, mode.width, mode.height, self.eye_w, (self.eye_w, eh),
                                 grid=GRID, depth_h=DEPTH_H, ss=S.ss)
        self.pipeline: Pipeline | None = None
        self.worker: DepthWorker | None = None
        self.panx = self.pany = 0.0
        self.fps = 0.0
        self.depth_hz = 0.0              # depth refreshes per second
        self._fps_t, self._fps_n, self._depth_n = time.monotonic(), 0, 0
        self._t0 = time.monotonic()     # base for the panel's shader animation time
        self._prof = ({k: 0.0 for k in _PROF_KEYS} | {"_n": 0}
                      if os.environ.get("BRAINDANCE_PROF") else None)

    # --- the depth path ------------------------------------------------------

    def _ensure_pipeline(self) -> None:
        """Build the pipeline for the current DEPTH settings, or rebuild it after
        a MODEL/TENSORRT change; then run it inline (SYNC) or on the worker."""
        S = self.S
        backend = _get_backend(S)
        if self.pipeline is None or self.pipeline.backend is not backend:
            self._stop_worker()
            stereo = _get_stereo()
            self.pipeline = Pipeline(backend, stereo)
            self.renderer.attach_depth(backend.in_size, stereo.size if stereo else (1, 1),
                                       stereo.uv if stereo else 0.0)
            h, w = backend.in_size
            log(f"depth: {backend.name}, {h}x{w}, "
                f"{'interop' if self.renderer.interop else 'readback'}"
                f"{', mlbw' if stereo else ''}")
        if self.pipeline.stereo is None and S.warp == "mlbw":
            log("no mlbw engines: using the mesh warp")
            S.warp = "mesh"
        if S.sync and self.worker is not None:
            self._stop_worker()
        elif not S.sync and self.worker is None:
            self.worker = DepthWorker(self.pipeline)
            self.worker.start()

    def _stop_worker(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.worker = None

    def _update_depth(self) -> None:
        S, r = self.S, self.renderer
        try:
            self._ensure_pipeline()
        except Exception as e:
            log(f"cannot load the depth model: {e}")
            S.live3d, S.depth_view = False, "off"
            return
        p = self.pipeline.params
        p.range_ema, p.stabilize, p.dilation = S.range_ema, S.stabilize, S.dilation
        p.stereo = S.warp == "mlbw" and (S.live3d or S.depth_view == "3d")
        p.disp, p.eye_w = S.disp, self.eye_w
        if self.worker is not None:
            if self.worker.want_frame():            # readback only when idle
                f = r.depth_frame()
                if f is not None:
                    self.worker.submit(f)
            res = self.worker.get()
        elif r.interop:
            # sync, zero-copy: blit -> CUDA reads it -> pipeline -> CUDA writes
            # the textures
            res = self.pipeline.from_gl(r.depth_src_gl())
        else:
            # sync: depth of this very frame (no 1-frame desync)
            res = self.pipeline.from_frame(r.depth_frame(use_pbo=False))
        if res is not None:
            r.set_result(res)
            self._depth_n += 1

    @property
    def depth_state(self) -> str:
        """For the panel and `braindance status`: OFF (no depth needed), IDLE (the
        picture holds still and its depth has settled), or refreshes per second."""
        if not (self.S.live3d or self.S.depth_view != "off") or self.pipeline is None:
            return "OFF"
        if self.pipeline.settled:
            return "IDLE"
        return f"{self.depth_hz:.0f}/s"

    # --- input and frames ----------------------------------------------------

    def zoom(self, steps: int) -> None:
        self.S.zoom = min(8.0, max(0.2, self.S.zoom * 1.1 ** steps))

    def pan(self, dx: float, dy: float) -> None:
        self.panx += dx
        self.pany += dy

    def reset_view(self) -> None:
        self.S.zoom, self.panx, self.pany = 1.0, 0.0, 0.0

    def _lap(self, key, t):
        if self._prof is None:
            return t
        n = time.perf_counter()
        self._prof[key] += n - t
        return n

    def frame(self, menu=None, opened_at=0.0, pointer=None, panel_xy=(0, 0)) -> None:
        """Draw one frame, with the settings panel when `menu` is given: its
        top-left at `panel_xy` (eye pixels), opened at `opened_at` (for its
        materialize animation), and the cursor on it at `pointer`. Blocks on
        vsync."""
        S, r = self.S, self.renderer
        ss = S.ss if S.warp == "mesh" else 1.0      # supersampling serves the mesh
        if ss != r.ss:
            r.set_ss(ss)

        t = time.perf_counter()
        windows = self.cap.update()
        t = self._lap("capture", t)
        r.composite(windows)
        t = self._lap("composite", t)

        if S.live3d or S.depth_view != "off":       # the depth views need a map too
            self._update_depth()
        t = self._lap("depth", t)

        # the 3D depth view warps with disparity too, so it shows parallax in FLAT
        disp = S.disp if (S.live3d or S.depth_view == "3d") else 0.0
        inpaint = None
        if disp and S.warp == "mesh" and S.edge_fill == "network":
            inpaint = _get_inpaint()
            if inpaint is None:
                log("no inpaint engine: using the mirror fill")
                S.edge_fill = "mirror"
        r.draw(disp, S.zoom, self.panx, self.pany,
               pointer=None if menu is not None else self.cap.pointer(), warp=S.warp,
               edge_fill=S.edge_fill, edge_cliff=S.edge_cliff, edge_halo=int(S.edge_halo),
               inpaint=inpaint,
               sharp_mode=S.sharp_mode, jbu_color=S.jbu_color,
               jbu_reach=S.jbu_reach, jbu_depth=S.jbu_depth,
               gf_eps=S.gf_eps, gf_radius=int(S.gf_radius),
               climit_px=S.climit, screen_depth_px=S.screen_depth,
               fill_debug=S.fill_debug, depth_view=S.depth_view)
        now = time.monotonic()
        if menu is not None:
            names, ti = menu.tab_bar()
            r.panel.update(names, ti, menu.rows(), self.fps, ("DEPTH", self.depth_state))
            r.draw_panel(panel_xy, now - self._t0, min(1.0, (now - opened_at) / 0.30),
                         S.screen_depth, pointer)
        t = self._lap("present", t)
        if self._prof is not None:
            self.stage.ctx.finish()      # GPU done, so the swap below = vsync idle
            t = self._lap("gpu_finish", t)

        self._fps_n += 1
        if now - self._fps_t >= 1.0:
            self.fps = self._fps_n / (now - self._fps_t)
            self.depth_hz = self._depth_n / (now - self._fps_t)
            self._fps_t, self._fps_n, self._depth_n = now, 0, 0
        self.stage.swap()
        self.stage.poll()
        if self._prof is not None:
            self._lap("swap", t)
            self._prof["_n"] += 1
            if self._prof["_n"] >= 120:
                self._dump_prof()

    def _dump_prof(self) -> None:
        p = self._prof
        n = max(1, p["_n"])
        line = "  ".join(f"{k}={p[k] / n * 1000:4.1f}" for k in _PROF_KEYS)
        tot = sum(p[k] for k in _PROF_KEYS) / n * 1000
        print(f"[prof] {line} | frame={tot:5.1f}ms {1000.0 / max(tot, 1e-3):.0f}fps", flush=True)
        for k in _PROF_KEYS:
            p[k] = 0.0
        p["_n"] = 0

    def close(self) -> None:
        self._stop_worker()
        self.renderer.release()
        self.cap.shutdown()             # X paints the desktop again
