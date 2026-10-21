"""presenter.py — the GL surfaces braindance draws on, one at a time.

  Stage        3D modes: a window as big as the glasses' whole side-by-side
               output. capture.cpp moves it into X's composite overlay window,
               above every other window, and maps it there, so no window manager
               ever handles it.
  PanelWindow  2D modes: X drives the glasses directly, and the settings panel is
               a small window of its own over the desktop, mapped while the
               panel is open.

Both let input through to the windows beneath. While the panel is open the
session holds the pointer itself and works the panel from where it points
(input.Controls), so the panel needs no input of its own.
"""
import os

import glfw
import moderngl
from Xlib import X, display as Xdisplay
from Xlib.ext import shape

from . import panel


def init() -> None:
    # No compositor runs under the live desktop, so have the NVIDIA driver sync
    # buffer swaps to vblank itself: it is read when the GL context is created.
    os.environ.setdefault("__GL_SYNC_TO_VBLANK", "1")
    if not glfw.init():
        raise RuntimeError("glfw init failed")


def terminate() -> None:
    glfw.terminate()


def _window(width: int, height: int, transparent: bool = False):
    glfw.default_window_hints()
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.DECORATED, glfw.FALSE)
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    glfw.window_hint(glfw.TRANSPARENT_FRAMEBUFFER, glfw.TRUE if transparent else glfw.FALSE)
    win = glfw.create_window(width, height, "braindance", None, None)
    if not win:
        raise RuntimeError("glfw window creation failed")
    glfw.make_context_current(win)
    glfw.swap_interval(1)
    return win


class Stage:
    def __init__(self, width: int, height: int):
        self.win = _window(width, height)
        self.xid = glfw.get_x11_window(self.win)
        # the X Display* (int address) for the capture lib: GLX pixmap binding
        # must be on the same connection as the context
        self.x_display = glfw.get_x11_display()
        self.ctx = moderngl.create_context()
        # Hide the hardware cursor: one sprite would show in one eye only; the
        # renderer draws a cursor into both. Held while this connection lives.
        self._xdpy = Xdisplay.Display()
        xwin = self._xdpy.create_resource_object("window", self.xid)
        xwin.change_attributes(override_redirect=True)
        self._xdpy.xfixes_query_version()
        xwin.xfixes_hide_cursor()
        self._xdpy.sync()

    def poll(self) -> None:
        glfw.poll_events()

    def swap(self) -> None:
        glfw.swap_buffers(self.win)

    def close(self) -> None:
        self._xdpy.close()               # shows the cursor again
        self.ctx.release()
        glfw.destroy_window(self.win)


class PanelWindow:
    def __init__(self):
        self.win = _window(panel.W, panel.H, transparent=True)
        self.ctx = moderngl.create_context()
        self.panel = panel.Panel(self.ctx)
        self._pos = (0, 0)
        self._xdpy = Xdisplay.Display()
        self._xwin = self._xdpy.create_resource_object("window", glfw.get_x11_window(self.win))
        self._xwin.change_attributes(override_redirect=True)
        self._xwin.shape_rectangles(shape.SO.Set, shape.SK.Input, 0, 0, 0, [])
        self._xdpy.sync()
        self.shown = False

    def show(self, xy) -> None:
        glfw.show_window(self.win)
        # after the map: glfw can move a window when it shows it
        self._pos = xy
        self._xwin.configure(x=xy[0], y=xy[1], stack_mode=X.Above)
        self._xdpy.sync()
        self.shown = True

    def move(self, xy) -> None:
        if xy != self._pos:
            self._pos = xy
            self._xwin.configure(x=xy[0], y=xy[1])
            self._xdpy.flush()

    def hide(self) -> None:
        glfw.hide_window(self.win)
        self.shown = False

    def draw(self, t: float, appear: float) -> None:
        glfw.make_context_current(self.win)
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)
        self.panel.draw(panel.FULL, t, appear)
        glfw.swap_buffers(self.win)
        glfw.poll_events()

    def close(self) -> None:
        self._xdpy.close()
        self.ctx.release()
        glfw.destroy_window(self.win)
