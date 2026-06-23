"""input.py — Super+Alt-gated controls via Xlib (a dedicated connection).

Keyboard read with XQueryKeymap (works regardless of focus); mouse via passive
XGrabButton on the Super+Alt chord so the combos are consumed (apps don't see
them) while bare clicks/scroll pass through. Mirrors stereo-mirror's scheme.

  Super+Alt + Q            quit
  Super+Alt + = / -        depth strength (disparity) + / -
  Super+Alt + scroll       zoom in / out
  Super+Alt + left-drag    pan
  Super+Alt + side btn 8   reset zoom + pan
  Super+Alt + side btn 9   toggle FLAT <-> DEPTH
"""
from types import SimpleNamespace
from Xlib import display as Xdisplay, X, XK


class Controls:
    def __init__(self, reset_btn=8, toggle_btn=9):
        self.d = Xdisplay.Display()
        self.root = self.d.screen().root
        self.reset_btn, self.toggle_btn = reset_btn, toggle_btn
        self._kc = {name: self.d.keysym_to_keycode(ks) for name, ks in {
            "super_l": XK.XK_Super_L, "super_r": XK.XK_Super_R,
            "alt_l": XK.XK_Alt_L, "alt_r": XK.XK_Alt_R,
            "q": XK.XK_q, "equal": XK.XK_equal, "minus": XK.XK_minus, "plus": XK.XK_plus,
            "p": XK.XK_p,
        }.items()}
        # passive button grabs on Super+Alt (+ lock-key variants)
        mod = X.Mod4Mask | X.Mod1Mask
        mask = X.ButtonPressMask | X.ButtonReleaseMask | X.PointerMotionMask
        for m in (0, X.LockMask, X.Mod2Mask, X.LockMask | X.Mod2Mask):
            for b in (1, 4, 5, self.reset_btn, self.toggle_btn):
                self.root.grab_button(b, mod | m, True, mask,
                                      X.GrabModeAsync, X.GrabModeAsync, 0, 0)
        self.d.sync()
        self._dragging = False
        self._lx = self._ly = 0
        self._pkeys = {}

    def _down(self, name):
        kc = self._kc[name]
        if not kc:
            return False
        km = self.d.query_keymap()
        return bool(km[kc >> 3] & (1 << (kc & 7)))

    def update(self, dt):
        ev = SimpleNamespace(quit=False, zoom_mult=1.0, pan_dx=0.0, pan_dy=0.0,
                             disp_delta=0.0, reset=False, toggle=False, screenshot=False)

        mod = (self._down("super_l") or self._down("super_r")) and \
              (self._down("alt_l") or self._down("alt_r"))

        # chord-gated edge: fires once per press while Super+Alt is held
        def cedge(name):
            d = mod and self._down(name)
            e = d and not self._pkeys.get(name, False)
            self._pkeys[name] = d
            return e

        if cedge("q"):
            ev.quit = True
        if cedge("p"):
            ev.screenshot = True
        e_eq, e_pl, e_mi = cedge("equal"), cedge("plus"), cedge("minus")
        if e_eq or e_pl:
            ev.disp_delta += 1.0      # +1 px per press
        if e_mi:
            ev.disp_delta -= 1.0

        # mouse (grabbed combos only fire while Super+Alt held)
        scroll = 0
        while self.d.pending_events():
            e = self.d.next_event()
            if e.type == X.ButtonPress:
                b = e.detail
                if b == 4:
                    scroll += 1
                elif b == 5:
                    scroll -= 1
                elif b == 1:
                    self._dragging = True
                    self._lx, self._ly = e.root_x, e.root_y
                elif b == self.reset_btn:
                    ev.reset = True
                elif b == self.toggle_btn:
                    ev.toggle = True
            elif e.type == X.ButtonRelease:
                if e.detail == 1:
                    self._dragging = False
            elif e.type == X.MotionNotify and self._dragging:
                ev.pan_dx += e.root_x - self._lx
                ev.pan_dy += e.root_y - self._ly
                self._lx, self._ly = e.root_x, e.root_y

        if scroll:
            ev.zoom_mult = 1.1 ** scroll
        return ev

    def shutdown(self):
        try:
            self.root.ungrab_button(X.AnyButton, X.AnyModifier)
            self.d.sync()
            self.d.close()
        except Exception:
            pass
