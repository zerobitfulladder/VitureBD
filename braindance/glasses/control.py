"""control.py — the glasses' control channel, as the session uses it.

VITURE's SDK runs in a helper process (helper.py): its teardown can hang inside
the library, and a process can be killed where a thread cannot. Requests are
one at a time with a timeout; a helper that stops answering is killed.

The glasses take a mode change only while they receive a picture in their
current mode. Without one they accept the first change and refuse every later
one until they are power-cycled. The session therefore keeps X driving the
glasses at their current mode before every set_mode.
"""
import json
import os
import queue
import subprocess
import sys
import threading

from . import sdk
from .modes import Mode


class GlassesError(RuntimeError):
    pass


def present() -> bool:
    """True if any VITURE USB device is plugged in."""
    return bool(sdk.usb_pids())


class Control:
    READY_TIMEOUT = 15.0
    REPLY_TIMEOUT = 6.0
    CLOSE_TIMEOUT = 8.0

    def __init__(self, on_mode, six_dof: bool = False):
        """Start the helper and wait for its SDK session, in 6DOF (cameras and
        VIO running) or 3DOF. on_mode(code) is called from a reader thread
        whenever the glasses report a mode."""
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            p for p in (sdk.lib_dir(), env.get("LD_LIBRARY_PATH")) if p)
        self.six_dof = six_dof
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "braindance.glasses.helper"] + (["--6dof"] if six_dof else []),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, env=env)
        self._on_mode = on_mode
        self._replies: queue.Queue[dict | None] = queue.Queue()
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self.info: dict = {}
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._log, daemon=True).start()
        if not self._ready.wait(self.READY_TIMEOUT) or "error" in self.info:
            self.close()
            raise GlassesError(self.info.get("error", "the glasses' SDK did not start"))

    def _read(self) -> None:
        for line in self._proc.stdout:
            msg = json.loads(line)
            event = msg.get("event")
            if event in ("ready", "error"):
                self.info = msg
                self._ready.set()
            elif event == "mode":
                self._on_mode(msg["code"])
            else:
                self._replies.put(msg)
        self._replies.put(None)             # the helper is gone
        self._ready.set()

    def _log(self) -> None:
        for line in self._proc.stderr:
            if not line.startswith("[CARINA]"):     # the SDK's own chatter
                print("[glasses]", line.rstrip(), file=sys.stderr, flush=True)

    def _request(self, **req) -> dict:
        with self._lock:
            if self._proc.poll() is not None:
                raise GlassesError("the glasses' SDK helper has exited")
            self._proc.stdin.write(json.dumps(req) + "\n")
            self._proc.stdin.flush()
            try:
                reply = self._replies.get(timeout=self.REPLY_TIMEOUT)
            except queue.Empty:
                self._proc.kill()
                raise GlassesError(f"{req['op']}: no answer from the glasses' SDK") from None
        if reply is None:
            raise GlassesError(f"{req['op']}: the glasses' SDK helper exited")
        if not reply["ok"]:
            raise GlassesError(reply["error"])
        return reply

    def set_mode(self, mode: Mode) -> None:
        self._request(op="set_mode", code=mode.code)

    def get_mode(self) -> int:
        return self._request(op="get_mode")["code"]

    def track(self, lead: float) -> str:
        """Start the head pose stream, predicting `lead` seconds ahead (or change
        the lead of the running one); returns its shared-memory path (pose.py)."""
        return self._request(op="track", lead=lead)["path"]

    def untrack(self) -> None:
        self._request(op="untrack")

    def close(self) -> None:
        if self._proc.poll() is None:
            try:
                self._proc.stdin.close()
            except OSError:
                pass
            try:
                self._proc.wait(self.CLOSE_TIMEOUT)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
