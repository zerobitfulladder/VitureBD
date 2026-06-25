"""overlay.py — the fullscreen GL surface on the glasses.

A glfw window made override-redirect via Xlib so it covers the whole 3840x1080
panel (bypassing qtile / fake_screens). Created hidden, made override-redirect
while unmapped, then shown — avoids the tile-then-fix race. Owns the moderngl
context and exposes the X Display* + window id for the capture lib and input.
"""
import glfw
import moderngl
from Xlib import display as Xdisplay, X as Xlib_X
from Xlib.ext import shape, xfixes


class Overlay:
    def __init__(self, panel=None, rect=None):
        # rect = (x, y, w, h) of the glasses output to cover. When the desktop head
        # sits at the origin and the glasses are placed to its right, the overlay
        # must be created at the glasses' x-offset (not 0,0). Falls back to the
        # primary monitor at 0,0 (legacy single-display) when rect is None.
        if not glfw.init():
            raise RuntimeError("glfw init failed")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.DECORATED, glfw.FALSE)
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)   # map after override-redirect

        if rect is not None:
            self._x, self._y, self.pw, self.ph = rect
        else:
            mode = glfw.get_video_mode(glfw.get_primary_monitor())
            self.pw, self.ph = panel or (mode.size.width, mode.size.height)
            self._x, self._y = 0, 0
        self.eye_w = self.pw // 2

        self.win = glfw.create_window(self.pw, self.ph, "braindance", None, None)
        if not self.win:
            glfw.terminate()
            raise RuntimeError("glfw window creation failed")
        glfw.make_context_current(self.win)

        # Override-redirect + cover the glasses output, while still unmapped.
        self.xdpy = Xdisplay.Display()
        self.xid = glfw.get_x11_window(self.win)
        self._xwin = self.xdpy.create_resource_object("window", self.xid)
        self._xwin.change_attributes(override_redirect=True)
        self._xwin.configure(x=self._x, y=self._y, width=self.pw, height=self.ph)
        self.xdpy.sync()
        glfw.show_window(self.win)
        # Re-assert geometry after map — glfw can move the window on show.
        self._xwin.configure(x=self._x, y=self._y, width=self.pw, height=self.ph,
                             stack_mode=Xlib_X.Above)
        # Empty input shape => mouse/keyboard pass THROUGH to the desktop behind.
        # Our controls come from XQueryKeymap + passive Super+Alt button grabs, so
        # they still work; only this makes the desktop usable under the overlay.
        self._xwin.shape_rectangles(shape.SO.Set, shape.SK.Input, 0, 0, 0, [])
        # Hide the hardware cursor (one sprite shows in only one eye); we draw our
        # own stereo cursor instead. Held for the connection's lifetime.
        try:
            self.xdpy.xfixes_query_version()
            self._xwin.xfixes_hide_cursor()
        except Exception as e:
            print("[braindance] could not hide HW cursor:", e, flush=True)
        self.xdpy.sync()
        glfw.swap_interval(1)

        self.ctx = moderngl.create_context()
        # X Display* (int address) for the C capture lib; reuse the GL display so
        # GLX pixmap binding is on the same connection as the context.
        self.x_display = glfw.get_x11_display()

    def should_close(self):
        return glfw.window_should_close(self.win)

    def poll(self):
        glfw.poll_events()

    def raise_top(self):
        # keep above any window that restacks above us (e.g. a fullscreen video)
        self._xwin.configure(stack_mode=Xlib_X.Above)

    def swap(self):
        glfw.swap_buffers(self.win)

    def terminate(self):
        try:
            self._xwin.xfixes_show_cursor()
            self.xdpy.sync()
        except Exception:
            pass
        try:
            glfw.terminate()
        except Exception:
            pass
