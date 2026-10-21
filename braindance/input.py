"""input.py — braindance's mouse controls, on an X connection of their own.

The keyboard belongs to the windows and the window manager: braindance never
reads or grabs it, except while a video plays.

  side button (forward, X button 9)   open / close the settings panel
  side button (back, X button 8)      recenter the desktop placed in space,
                                      while it is (the apps have it otherwise)

Only the bare buttons are braindance's: with Super (or any other modifier but
Caps / Num Lock) held they stay the window manager's.

While the panel is open the pointer is braindance's alone (an active grab), so
no window sees a click or a scroll; the session decides what each one does
(session.py). While a video plays the keyboard and the pointer are both
grabbed, so every key and click reaches the player.

update() drains the connection's events into the panel and recenter buttons,
presses, releases, motion and, while the keyboard is grabbed, keys.
"""
from types import SimpleNamespace

from Xlib import X, XK, display as Xdisplay

# The forward side button opens the panel; the back one recenters, and only
# while there is something to recenter.
PANEL_BUTTON = 9
RECENTER_BUTTON = 8
# The bare button, whatever the lock keys: a grab on any modifier would clash
# with the window manager's own (qtile: Super + side buttons resize).
_LOCKS = (0, X.LockMask, X.Mod2Mask, X.LockMask | X.Mod2Mask)

# X keysym name -> the player's key name (letters and digits map to themselves)
_KEY_NAMES = {
    "Return": "return", "KP_Enter": "return", "Escape": "escape", "BackSpace": "backspace",
    "Tab": "tab", "ISO_Left_Tab": "tab", "Left": "left", "Right": "right", "Up": "up",
    "Down": "down", "Prior": "pageup", "Next": "pagedown", "Home": "home", "End": "end",
    "minus": "minus", "equal": "equal", "KP_Subtract": "minus", "KP_Add": "equal",
    "plus": "equal", "bracketleft": "bracketleft", "bracketright": "bracketright",
    "braceleft": "bracketleft", "braceright": "bracketright", "space": "space",
}
_POINTER_EVENTS = X.ButtonPressMask | X.ButtonReleaseMask | X.PointerMotionMask


class Controls:
    def __init__(self):
        self.d = Xdisplay.Display()
        self.root = self.d.screen().root
        # passive: the bare panel button reaches us, never the windows
        errors = []
        for mods in _LOCKS:
            self.root.grab_button(PANEL_BUTTON, mods, False,
                                  X.ButtonPressMask | X.ButtonReleaseMask,
                                  X.GrabModeAsync, X.GrabModeAsync, X.NONE, X.NONE,
                                  onerror=errors.append)
        self.d.sync()
        if errors:
            raise RuntimeError(f"another program holds mouse button {PANEL_BUTTON}: "
                               "braindance needs it for its panel")
        self._want = (False, False, False)    # (pointer, keyboard, recenter) asked for
        self._held = (False, False, False)    # what we hold
        self.refused: str | None = None       # why the recenter button could not be had

    # --- grabs ---------------------------------------------------------------

    def want(self, pointer: bool, keyboard: bool, recenter: bool = False) -> None:
        """Hold the pointer and/or the keyboard, and the recenter button. A
        pointer or keyboard grab someone else holds (an open menu, say) is
        retried on every update until it succeeds; a recenter button another
        program holds is given up (see `refused`) until it is asked for anew."""
        if recenter and not self._want[2]:
            self.refused = None
        self._want = (pointer, keyboard, recenter)
        self._sync()

    def _sync(self) -> None:
        pointer, keyboard, recenter = self._want
        held_p, held_k, held_r = self._held
        if recenter and not held_r and self.refused is None:
            errors = []
            for mods in _LOCKS:
                self.root.grab_button(RECENTER_BUTTON, mods, False,
                                      X.ButtonPressMask | X.ButtonReleaseMask,
                                      X.GrabModeAsync, X.GrabModeAsync, X.NONE, X.NONE,
                                      onerror=errors.append)
            self.d.sync()
            held_r = not errors
            if errors:
                self.refused = f"another program holds mouse button {RECENTER_BUTTON}"
                for mods in _LOCKS:
                    self.root.ungrab_button(RECENTER_BUTTON, mods)
        elif held_r and not recenter:
            for mods in _LOCKS:
                self.root.ungrab_button(RECENTER_BUTTON, mods)
            held_r = False
        if keyboard and not held_k:
            held_k = self.root.grab_keyboard(False, X.GrabModeAsync, X.GrabModeAsync,
                                             X.CurrentTime) == X.GrabSuccess
        elif held_k and not keyboard:
            self.d.ungrab_keyboard(X.CurrentTime)
            held_k = False
        if pointer and not held_p:
            held_p = self.root.grab_pointer(False, _POINTER_EVENTS, X.GrabModeAsync,
                                            X.GrabModeAsync, X.NONE, X.NONE,
                                            X.CurrentTime) == X.GrabSuccess
        elif held_p and not pointer:
            self.d.ungrab_pointer(X.CurrentTime)
            held_p = False
        self._held = (held_p, held_k, held_r)
        self.d.sync()

    def pointer(self) -> tuple[int, int]:
        p = self.root.query_pointer()
        return p.root_x, p.root_y

    def keep_awake(self) -> None:
        """Reset the screen saver / DPMS idle timer (a video has no input)."""
        self.d.force_screen_saver(X.ScreenSaverReset)
        self.d.flush()

    # --- events --------------------------------------------------------------

    def _key_name(self, e):
        shift = bool(e.state & X.ShiftMask)
        ks = self.d.keycode_to_keysym(e.detail, 1 if shift else 0) or \
            self.d.keycode_to_keysym(e.detail, 0)
        name = XK.keysym_to_string(ks) if ks else None
        if name is None:
            return None
        if name in _KEY_NAMES:
            return _KEY_NAMES[name]
        if len(name) == 1 and name.isalnum():
            return name.lower()
        return None

    def update(self):
        """Everything since the last update: panel (the panel button was
        pressed), recenter (the recenter button was, while it is wanted),
        presses [(button, x, y, shift)] and releases [(button, x, y)], motion
        (x, y) or None, and keys [(name, shift)]. Positions are root pixels."""
        if self._held != self._want and (self._want[:2] != self._held[:2] or self.refused is None):
            self._sync()
        ev = SimpleNamespace(panel=False, recenter=False, presses=[], releases=[], motion=None,
                             keys=[])
        ours = {PANEL_BUTTON} | ({RECENTER_BUTTON} if self._want[2] else set())
        while self.d.pending_events():
            e = self.d.next_event()
            if e.type == X.ButtonPress:
                if e.detail == PANEL_BUTTON:
                    ev.panel = not ev.panel
                elif e.detail in ours:
                    ev.recenter = True
                else:
                    ev.presses.append((e.detail, e.root_x, e.root_y, bool(e.state & X.ShiftMask)))
            elif e.type == X.ButtonRelease:
                if e.detail not in ours:
                    ev.releases.append((e.detail, e.root_x, e.root_y))
            elif e.type == X.MotionNotify:
                ev.motion = (e.root_x, e.root_y)
            elif e.type == X.KeyPress:
                name = self._key_name(e)
                if name:
                    ev.keys.append((name, bool(e.state & X.ShiftMask)))
        return ev

    def shutdown(self) -> None:
        try:
            self.want(False, False, False)
            for mods in _LOCKS:
                self.root.ungrab_button(PANEL_BUTTON, mods)
            self.d.sync()
            self.d.close()
        except Exception:
            pass
