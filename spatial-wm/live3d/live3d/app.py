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

_STATE = os.path.expanduser("~/.config/live3d/state.json")


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

from .overlay import Overlay
from .capture import DesktopCapture
from .render import Renderer
from .input import Controls
from .depth import make_depth


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
                print("[live3d] depth error:", e, flush=True)
                d = None
            with self._lock:
                if d is not None:
                    self._result = d
                self._busy = False

    def stop(self):
        self._stop = True
        self._ev.set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="depth-anything/Depth-Anything-V2-Small-hf")
    ap.add_argument("--in-h", type=int, default=392, help="depth model input height")
    ap.add_argument("--depth-h", type=int, default=392, help="readback height for depth")
    ap.add_argument("--disp", type=float, default=18.0, help="max disparity px (3D strength)")
    ap.add_argument("--conv", type=float, default=0.5, help="convergence depth 0..1")
    ap.add_argument("--grid", type=int, default=256)
    ap.add_argument("--mode", choices=["flat", "depth"], default="flat")
    ap.add_argument("--trt", action="store_true", help="use TensorRT FP16 depth (DA2, fixed 392x686)")
    ap.add_argument("--panel", default=None, help="WxH override, e.g. 3840x1080")
    ap.add_argument("--sync", action="store_true", help="compute depth inline (debug)")
    args = ap.parse_args()

    panel = None
    if args.panel:
        panel = tuple(int(v) for v in args.panel.split("x"))

    ov = Overlay(panel=panel)
    region = (0, 0, ov.eye_w, ov.ph)
    print(f"[live3d] panel {ov.pw}x{ov.ph}  eye {ov.eye_w}x{ov.ph}  region {region}", flush=True)

    cap = DesktopCapture(ov.ctx, ov.x_display, ov.xid, region)
    renderer = Renderer(ov.ctx, ov.pw, ov.ph, ov.eye_w, (ov.eye_w, ov.ph),
                        grid=args.grid, depth_h=args.depth_h)
    print("[live3d] loading depth model...", flush=True)
    depth = make_depth(args)
    controls = Controls()

    worker = None
    if not args.sync:
        worker = DepthWorker(depth)
        worker.start()

    saved = load_state()
    mode = args.mode                      # NOT persisted — predictable start (FLAT)
    disparity = float(saved.get("disp", args.disp))
    conv = float(saved.get("conv", args.conv))
    zoom, panx, pany = 1.0, 0.0, 0.0

    fps_t, fps_n, fps, tick = time.time(), 0, 0.0, 0
    try:
        while not ov.should_close():
            now = time.time()
            dt = max(1e-3, now - getattr(main, "_last", now))
            main._last = now
            tick += 1
            ov.poll()

            ev = controls.update(dt)
            if ev.quit:
                break
            zoom = min(8.0, max(0.2, zoom * ev.zoom_mult))
            panx += ev.pan_dx
            pany += ev.pan_dy
            disparity = max(0.0, disparity + ev.disp_delta)  # +/- 1 px per press
            if ev.reset:
                zoom, panx, pany = 1.0, 0.0, 0.0
            if ev.toggle:
                mode = "depth" if mode == "flat" else "flat"
            if ev.disp_delta:
                save_state({"disp": disparity, "conv": conv})

            windows = cap.update()
            sig = tuple((w[1], w[2], w[3], w[4]) for w in windows)   # x,y,w,h per window
            if sig != getattr(main, "_winsig", None):
                main._winsig = sig
                # A restack (fullscreen toggle, new/closed window) can lift a real
                # window above our override-redirect overlay; since Composite is in
                # Automatic mode the server then paints that raw window on top,
                # occluding the warped scene + cursor. Reclaim top over the next few
                # frames — the WM's restack and resize don't land in one frame.
                main._raise_until = tick + 10
                print(f"[live3d] windows changed ({len(windows)}): "
                      + ", ".join(f"{w}x{h}@{x},{y}" for (_, x, y, w, h) in windows), flush=True)
            if tick < getattr(main, "_raise_until", 0) or tick % 60 == 1:
                ov.raise_top()         # event-driven; periodic % 60 is the backstop
            renderer.composite(windows)

            if mode == "depth":
                if worker is not None:
                    if worker.want_frame():                  # readback only when idle
                        f = renderer.depth_frame()
                        if f is not None:
                            worker.submit(f)
                    d = worker.get()
                    if d is not None:
                        renderer.set_depth(d)
                else:
                    # sync: read the current frame (matched depth, no 1-frame desync)
                    renderer.set_depth(depth(renderer.depth_frame(use_pbo=False)))

            renderer.draw(disparity if mode == "depth" else 0.0, conv, zoom, panx, pany,
                          pointer=cap.pointer())

            if ev.screenshot:
                from PIL import Image
                data = ov.ctx.screen.read(viewport=(ov.pw, ov.ph), components=3)
                Image.frombytes("RGB", (ov.pw, ov.ph), data).transpose(
                    Image.FLIP_TOP_BOTTOM).save("/home/lavender/live3d_grab.png")
                print("[live3d] saved /home/lavender/live3d_grab.png", flush=True)

            fps_n += 1
            if now - fps_t >= 1.0:
                fps = fps_n / (now - fps_t)
                fps_t, fps_n = now, 0
                print(f"[live3d] {mode}  {fps:.0f} fps  disp {disparity:.0f}  zoom {zoom:.2f}",
                      flush=True)
            ov.swap()
    finally:
        save_state({"disp": disparity, "conv": conv})
        if worker is not None:
            worker.stop()
        controls.shutdown()
        cap.shutdown()
        ov.terminate()


if __name__ == "__main__":
    main()
