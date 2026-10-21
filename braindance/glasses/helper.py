"""helper.py — the glasses' SDK session, in a process of its own.

control.py starts this as `python -m braindance.glasses.helper [--6dof]` and
speaks JSON lines with it: requests on stdin, replies and events on stdout.

  -> {"op": "set_mode", "code": 50}    <- {"ok": true}
  -> {"op": "get_mode"}                <- {"ok": true, "code": 50}
  -> {"op": "track", "lead": 0.012}    <- {"ok": true, "path": "/dev/shm/..."}
  -> {"op": "untrack"}                 <- {"ok": true}
                                       <- {"ok": false, "error": "..."}
  unsolicited                          <- {"event": "ready", "pid": ..., "name": ..., "firmware": ...}
                                       <- {"event": "mode", "code": 50}
                                       <- {"event": "error", "error": "..."}

"mode" events come from the glasses: their own 2D/3D button, or the result of a
set_mode. "track" starts the pose stream (pose.py), predicting `lead` seconds
ahead, or changes the lead of a running one. --6dof opens the SDK with the
cameras and VIO, for position as well as orientation. Closing stdin ends the
session.
"""
import json
import os
import sys
import threading
import time

from .pose import PoseWriter
from .sdk import CALLBACK_DISPLAY_MODE, Glasses, SdkError

# The SDK's stop() can hang if it runs within about a second of start(), so a
# session stays open at least this long.
MIN_LIFETIME = 3.0

# libglasses prints its own log lines to stdout. Keep the protocol on a private
# copy of the pipe and send fd 1 (C stdio included) to stderr.
_proto = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)
sys.stdout = sys.stderr
_lock = threading.Lock()


def _send(msg: dict) -> None:
    with _lock:
        _proto.write(json.dumps(msg) + "\n")


def _on_state(state_id: int, value: int) -> None:
    if state_id == CALLBACK_DISPLAY_MODE:
        _send({"event": "mode", "code": value})


class _PoseStream(threading.Thread):
    """Polls the SDK for the predicted head pose and keeps the latest in shared
    memory, about every millisecond."""
    PERIOD = 0.001

    def __init__(self, g: Glasses, lead: float):
        super().__init__(daemon=True)
        self.g, self.lead = g, lead
        self.writer = PoseWriter(f"/dev/shm/braindance-pose-{os.getpid()}")
        self._quit = threading.Event()

    def run(self) -> None:
        while not self._quit.wait(self.PERIOD):
            try:
                pose, status = self.g.pose(self.lead)
            except SdkError:
                continue
            self.writer.write(time.monotonic(), pose, status)

    def stop(self) -> None:
        self._quit.set()
        self.join()
        self.writer.close()


_stream: _PoseStream | None = None


def _handle(g: Glasses, req: dict) -> dict:
    global _stream
    op = req.get("op")
    if op == "set_mode":
        g.set_mode(int(req["code"]))
        return {"ok": True}
    if op == "get_mode":
        return {"ok": True, "code": g.get_mode()}
    if op == "track":
        if _stream is None:
            _stream = _PoseStream(g, float(req["lead"]))
            _stream.start()
        _stream.lead = float(req["lead"])
        return {"ok": True, "path": _stream.writer.path}
    if op == "untrack":
        if _stream is not None:
            _stream.stop()
            _stream = None
        return {"ok": True}
    return {"ok": False, "error": f"unknown op {op!r}"}


def main() -> int:
    try:
        g = Glasses(_on_state, six_dof="--6dof" in sys.argv[1:])
    except (RuntimeError, OSError) as e:
        _send({"event": "error", "error": str(e)})
        return 1
    opened = time.monotonic()
    try:
        firmware = g.firmware
    except SdkError:
        firmware = None
    _send({"event": "ready", "pid": g.pid, "name": g.name, "firmware": firmware})
    try:
        for line in sys.stdin:
            try:
                reply = _handle(g, json.loads(line))
            except SdkError as e:
                reply = {"ok": False, "error": str(e)}
            _send(reply)
    finally:
        if _stream is not None:
            _stream.stop()
        time.sleep(max(0.0, MIN_LIFETIME - (time.monotonic() - opened)))
        g.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
