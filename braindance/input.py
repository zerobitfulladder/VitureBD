"""input.py — Super+Alt-gated controls via Xlib (a dedicated connection).

Keyboard read with XQueryKeymap (works regardless of focus); mouse via passive
XGrabButton on the Super+Alt chord so the combos are consumed (apps don't see
them) while bare clicks/scroll pass through. Mirrors stereo-mirror's scheme.

  Super+Alt + S            toggle settings panel (OSD)
  Super+Alt + Q            quit
  Super+Alt + = / -        depth strength (disparity) + / -
  Super+Alt + [ / ]        screen depth (plane in / out) - / +
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
            "s": XK.XK_s,
            "bracketleft": XK.XK_bracketleft, "bracketright": XK.XK_bracketright,
            # menu navigation (bare keys, grabbed only while the panel is open)
            "up": XK.XK_Up, "down": XK.XK_Down, "left": XK.XK_Left,
            "right": XK.XK_Right, "return": XK.XK_Return, "tab": XK.XK_Tab,
            "shift_l": XK.XK_Shift_L, "shift_r": XK.XK_Shift_R,
        }.items()}
        self._menu_keys = ("up", "down", "left", "right", "return", "tab")
        self._menu_grabbed = False
        self._lockvariants = (0, X.LockMask, X.Mod2Mask, X.LockMask | X.Mod2Mask)
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

    def grab_menu_keys(self):
        """Grab the bare arrow/Enter keys so the panel is modal — the focused app
        won't see them while we navigate. Call when the panel opens."""
        if self._menu_grabbed:
            return
        for name in self._menu_keys:
            kc = self._kc[name]
            if not kc:
                continue
            for m in self._lockvariants:
                try:
                    self.root.grab_key(kc, m, False, X.GrabModeAsync, X.GrabModeAsync)
                except Exception:
                    pass
        # also grab Shift+Tab (reverse tab) so the focused app never sees it
        tkc = self._kc["tab"]
        if tkc:
            for m in self._lockvariants:
                try:
                    self.root.grab_key(tkc, X.ShiftMask | m, False,
                                       X.GrabModeAsync, X.GrabModeAsync)
                except Exception:
                    pass
        self.d.sync()
        self._menu_grabbed = True

    def ungrab_menu_keys(self):
        if not self._menu_grabbed:
            return
        for name in self._menu_keys:
            kc = self._kc[name]
            if not kc:
                continue
            for m in self._lockvariants:
                try:
                    self.root.ungrab_key(kc, m)
                except Exception:
                    pass
        tkc = self._kc["tab"]
        if tkc:
            for m in self._lockvariants:
                try:
                    self.root.ungrab_key(tkc, X.ShiftMask | m)
                except Exception:
                    pass
        self.d.sync()
        self._menu_grabbed = False

    def update(self, dt):
        ev = SimpleNamespace(quit=False, zoom_mult=1.0, pan_dx=0.0, pan_dy=0.0,
                             disp_delta=0.0, screen_delta=0.0, reset=False, toggle=False,
                             panel=False,
                             nav_up=False, nav_down=False, nav_left=False,
                             nav_right=False, enter=False,
                             tab_next=False, tab_prev=False)

        mod = (self._down("super_l") or self._down("super_r")) and \
              (self._down("alt_l") or self._down("alt_r"))

        # chord-gated edge: fires once per press while Super+Alt is held
        def cedge(name):
            d = mod and self._down(name)
            e = d and not self._pkeys.get(name, False)
            self._pkeys[name] = d
            return e

        # bare edge: fires once per press regardless of modifiers (menu nav). The
        # app only consults these while the panel is open (and the keys are grabbed).
        def bedge(name):
            d = self._down(name)
            e = d and not self._pkeys.get(name, False)
            self._pkeys[name] = d
            return e

        ev.nav_up = bedge("up")
        ev.nav_down = bedge("down")
        ev.nav_left = bedge("left")
        ev.nav_right = bedge("right")
        ev.enter = bedge("return")
        # Tab / Shift+Tab switch panel tabs (shift read live, like the chord mods)
        if bedge("tab"):
            if self._down("shift_l") or self._down("shift_r"):
                ev.tab_prev = True
            else:
                ev.tab_next = True

        if cedge("q"):
            ev.quit = True
        if cedge("s"):
            ev.panel = True
        e_eq, e_pl, e_mi = cedge("equal"), cedge("plus"), cedge("minus")
        if e_eq or e_pl:
            ev.disp_delta += 1.0      # +1 px per press
        if e_mi:
            ev.disp_delta -= 1.0
        if cedge("bracketright"):
            ev.screen_delta += 1.0    # screen depth: plane closer (+1 px/press)
        if cedge("bracketleft"):
            ev.screen_delta -= 1.0

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
