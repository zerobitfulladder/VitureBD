"""ipc.py — the running session's control socket.

A session listens on $XDG_RUNTIME_DIR/braindance/session.sock. Each connection
carries one JSON request line and gets one JSON reply line. The server thread
only accepts and parses; the session answers from its own thread through the
reply callback it is handed.
"""
import json
import os
import socket
import threading


def path() -> str:
    return os.path.join(os.environ.get("XDG_RUNTIME_DIR", "/tmp"), "braindance", "session.sock")


class NotRunning(RuntimeError):
    pass


def request(req: dict, timeout: float = 60.0) -> dict:
    """Send one request to the running session and return its reply."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(path())
    except (FileNotFoundError, ConnectionRefusedError):
        s.close()
        raise NotRunning("no braindance session is running") from None
    with s, s.makefile("rw") as f:
        f.write(json.dumps(req) + "\n")
        f.flush()
        line = f.readline()
    if not line:
        raise RuntimeError("the session closed the connection without answering")
    return json.loads(line)


def running() -> bool:
    try:
        request({"cmd": "status"}, timeout=2.0)
    except (NotRunning, OSError):
        return False
    return True


class Server:
    def __init__(self, on_request):
        """on_request(req, reply) is called from the server thread; the session
        calls reply(dict) once, from wherever it handles the request."""
        p = path()
        os.makedirs(os.path.dirname(p), exist_ok=True)
        if os.path.exists(p):
            if running():
                raise RuntimeError("a braindance session is already running")
            os.remove(p)             # left behind by a session that crashed
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.bind(p)
        self._sock.listen(4)
        self._on_request = on_request
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self) -> None:
        while True:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return               # closed
            f = conn.makefile("rw")
            try:
                req = json.loads(f.readline() or "null")
            except ValueError:
                req = None
            if not isinstance(req, dict):
                f.close()
                conn.close()
                continue

            def reply(msg: dict, f=f, conn=conn) -> None:
                try:
                    f.write(json.dumps(msg) + "\n")
                    f.flush()
                except OSError:
                    pass             # the client gave up waiting
                finally:
                    f.close()
                    conn.close()

            self._on_request(req, reply)

    def close(self) -> None:
        self._sock.close()
        try:
            os.remove(path())
        except FileNotFoundError:
            pass
