"""capture.py — ctypes bridge to libdesktopcapture.so (capture/).

Becomes the compositor: takes over the screen with the caller's window, and
captures each top-level window's backing pixmap as a GL texture in that
window's context, wrapped as a moderngl external texture so render.py can
composite them. No screen-grab, so no capture/display feedback.
"""
import ctypes
import os


class CapWin(ctypes.Structure):
    _fields_ = [
        ("xid", ctypes.c_ulong),
        ("tex", ctypes.c_uint),
        ("x", ctypes.c_int), ("y", ctypes.c_int),
        ("w", ctypes.c_int), ("h", ctypes.c_int),
    ]


_LIB = os.path.join(os.path.dirname(__file__), "..", "capture", "build", "libdesktopcapture.so")


class DesktopCapture:
    def __init__(self, ctx, x_display, self_xid, region):
        lib = ctypes.CDLL(os.path.abspath(_LIB))
        lib.cap_init.argtypes = [ctypes.c_void_p, ctypes.c_ulong,
                                 ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.cap_init.restype = ctypes.c_int
        lib.cap_count.restype = ctypes.c_int
        lib.cap_windows.restype = ctypes.POINTER(CapWin)
        lib.cap_pointer.argtypes = [ctypes.POINTER(ctypes.c_float),
                                    ctypes.POINTER(ctypes.c_float),
                                    ctypes.POINTER(ctypes.c_int)]
        self.lib = lib
        self.ctx = ctx
        self.rx, self.ry, self.rw, self.rh = region
        if not lib.cap_init(ctypes.c_void_p(int(x_display)), ctypes.c_ulong(int(self_xid)),
                            self.rx, self.ry, self.rw, self.rh):
            raise RuntimeError("cap_init failed: is another compositor running?")
        self._ext = {}   # GL texture name -> (moderngl external texture, (w, h))

    def update(self):
        """Returns [(moderngl_texture, x, y, w, h), ...] in region-local coords."""
        self.lib.cap_update()
        n = self.lib.cap_count()
        wins = self.lib.cap_windows()
        out = []
        for i in range(n):
            cw = wins[i]
            cached = self._ext.get(cw.tex)
            if cached is None or cached[1] != (cw.w, cw.h):
                tex = self.ctx.external_texture(cw.tex, (cw.w, cw.h), 3, 0, "f1")
                self._ext[cw.tex] = (tex, (cw.w, cw.h))
            else:
                tex = cached[0]
            out.append((tex, cw.x - self.rx, cw.y - self.ry, cw.w, cw.h))
        return out

    def pointer(self):
        """(u, v, inside) — pointer normalized in the region."""
        u, v, inside = ctypes.c_float(), ctypes.c_float(), ctypes.c_int()
        self.lib.cap_pointer(ctypes.byref(u), ctypes.byref(v), ctypes.byref(inside))
        return u.value, v.value, bool(inside.value)

    def shutdown(self):
        self.lib.cap_shutdown()
