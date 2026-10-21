"""session.py — one braindance session: it owns the glasses and the X outputs
from start to exit.

  start  The glasses must be plugged in. The current layout is saved (under
         $XDG_RUNTIME_DIR, so `braindance restore` can put it back after a
         crash), the braindance monitor is published, and the glasses are lit
         as the only output, in the mode they are in.
  run    In a 2D mode X drives the glasses directly, unless SPACE MODE places
         the desktop in space by the glasses' tracking (space.py), which takes
         over the screen. In a 3D (side-by-side) mode the live desktop
         (desktop.py) takes over the screen: FLAT, or LIVE 3D. The settings
         panel (the mouse's forward side button) switches modes, as do
         `braindance mode NAME` and the glasses' own 2D/3D button.
         Unplugging the glasses ends the session, as do END SESSION on the
         panel and `braindance quit`.
  play   `braindance play FILE` plays a 3D video (media.py), and `braindance
         view FILE` shows a 3D model placed in the room (viewer.py), on the
         stage the live desktop draws on, which pauses meanwhile. From a 2D
         mode the glasses switch to 3D for it, and back when it ends.
  exit   The glasses go back to the mode they started in, the monitor is
         withdrawn, and the saved layout is put back.

Everything runs on the main thread, one frame at a time: the IPC server and
the glasses' helper only post events to a queue the loop drains.
"""
import os
import queue
import signal
import subprocess
import time

from . import displays, ipc, panel, presenter, settings
from .desktop import Desktop
from .glasses import modes
from .glasses.control import Control, GlassesError, present
from .glasses.modes import Mode
from .input import Controls
from .media import Player
from .menu import Section, Setting
from .space import Space
from .viewer import MODEL_TYPES, ModelView, is_model

# After a mode change, the glasses take another one only once they have locked
# onto a picture in the new mode.
SETTLE = 2.0
# The glasses drop off DisplayPort and come back with the new mode in about 3 s.
EDID_TIMEOUT = 10.0
SCAN_TIMEOUT = 5.0
IDLE = 0.05              # loop period in 2D with the panel closed
PANEL_APPEAR = 0.30      # the panel's materialize animation, seconds
AWAKE = 30.0             # while a video plays, reset the screen saver this often


class SessionError(RuntimeError):
    pass


class Fatal(SessionError):
    """The session cannot go on safely: end it and restore the displays."""


def log(*args) -> None:
    print("[braindance]", *args, flush=True)


class Session:
    def __init__(self):
        """Find the glasses and the mode they are in. Changes nothing yet."""
        if not present():
            raise SessionError("no VITURE glasses plugged in")
        self._found = displays.query()
        g = displays.glasses(self._found)
        if g is None:
            raise SessionError("the glasses are on USB, but X sees no VITURE display")
        timing = g.preferred
        mode = modes.by_timing(*timing) if timing else None
        if timing is None or mode is None:
            raise SessionError(f"the glasses offer {timing}, which braindance does not "
                               "drive; replug them")
        self.output = g.name
        self.mode = self.start_mode = self.pending = mode
        self._start_timing = timing
        self.events: queue.Queue[tuple] = queue.Queue()
        self.control: Control | None = None
        self.controls: Controls | None = None
        self.snapshot: list[dict] = []
        self.running = False

        self.S = settings.load()
        self._saved = settings.signature(self.S)
        self.menu = settings.build_menu(self.S, self._glasses_tab(), lambda: self.mode.sbs,
                                        self._recenter, lambda: isinstance(self.app, ModelView))
        self.panel_open = False
        self.panel_opened_at = 0.0
        self.pointer = (0, 0)            # root pixels, while the panel is open
        self._drag: tuple[int, int] | None = None     # a pan in progress
        self._panel_grab: tuple[int, int] | None = None   # where a moving panel is held
        self._panel_moving: tuple[int, int] | None = None  # its top-left meanwhile
        self.stage: presenter.Stage | None = None        # what the views below draw on
        self.desktop: Desktop | None = None               # 3D modes
        self.space: Space | None = None                   # 2D modes with SPACE MODE on
        self._lead = None                                 # the pose prediction asked for, s
        self.app: Player | ModelView | None = None        # a video or a model, over the 3D view
        self._resume: tuple[str, float] | None = None     # an app to (re)start in the next 3D view
        self._app_return: Mode | None = None              # the 2D mode an app took us from
        self._awake_at = 0.0
        self.panel_window: presenter.PanelWindow | None = None
        self._picom: list[str] | None = None     # the picom we stopped, to restart
        self._t0 = time.monotonic()

    # --- the GLASSES tab -----------------------------------------------------

    def _glasses_tab(self) -> list:
        return [
            Section("DISPLAY MODE"),
            Setting("OUTPUT", "enum", lambda: self.pending.name,
                    lambda v: setattr(self, "pending", modes.by_name(v)),
                    options=[m.name for m in modes.MODES],
                    fmt=lambda v: v.upper() + ("  ⏎" if v != self.mode.name else ""),
                    on_click=self._apply_pending,
                    help="The glasses' display mode. The wheel picks one, a click "
                         "switches to it; the glasses blank for a few seconds."),
            Section("SESSION"),
            Setting("END SESSION", "action", lambda: None, None, fmt=lambda v: "CLICK",
                    on_click=lambda: self.events.put(("quit",)),
                    help="Put the glasses back in the mode they started in and the "
                         "displays back as they were."),
        ]

    def _apply_pending(self) -> None:
        if self.pending != self.mode:
            self.events.put(("switch", self.pending))

    def _recenter(self) -> None:
        if isinstance(self.app, ModelView):
            self.app.recenter()
        elif self.space is not None:
            self.space.reset_view()

    # --- start and stop ------------------------------------------------------

    def start(self) -> None:
        # A snapshot left behind is the layout from before a session that
        # crashed: that is the one to go back to, not the glasses-only layout
        # it left lit.
        self.snapshot = displays.load_snapshot() or displays.snapshot(self._found)
        displays.save_snapshot(self.snapshot)
        self.running = True
        self._light(self.mode, self._start_timing)
        self.control = self._open_control(six_dof=self._space_wanted()[1])
        info = self.control.info
        log(f"{info['name']} (firmware {info['firmware']}) on {self.output}, {self.mode.name}")
        presenter.init()
        self.controls = Controls()
        self._enter_view()

    def stop(self) -> None:
        self._leave_view()
        if self.controls is not None:
            self.controls.shutdown()
            self.controls = None
        presenter.terminate()
        if self.control is not None:
            if self.mode != self.start_mode and self._glasses_connected():
                try:
                    self._retune(self.start_mode, send=True)
                except (SessionError, GlassesError) as e:
                    log(f"could not put the glasses back in {self.start_mode.name}: {e}")
            self.control.close()
            self.control = None
        # Withdraw the monitor before the outputs change: qtile and indigoshell
        # read it when they handle the RandR event the restore raises.
        displays.withdraw()
        displays.restore(self.snapshot)
        displays.forget_snapshot()
        self._start_picom()          # after the restore, against the final layout
        self.running = False
        log("displays restored")

    # --- views: what a mode shows ----------------------------------------------

    def _space_wanted(self) -> tuple[bool, bool]:
        """Whether this mode shows the desktop placed in space, and with 6DOF."""
        on = not self.mode.sbs and self.S.space_mode != "off"
        return on, on and self.S.space_dof == "6dof"

    def _enter_view(self) -> None:
        """3D: the live desktop takes over the screen, and an app waiting to
        start does. 2D: the desktop placed in space with SPACE MODE on, else X
        shows it. The glasses' SDK runs 6DOF only while the space view or a
        model uses it."""
        space, six_dof = self._space_wanted()
        if self.mode.sbs and self._resume is not None and is_model(self._resume[0]):
            six_dof = self.S.model_dof == "6dof"
        self._ensure_control(six_dof)
        if self.mode.sbs:
            self._stop_picom()               # the capture redirects every window itself
            try:
                self.stage = presenter.Stage(self.mode.width, self.mode.height)
                self.desktop = Desktop(self.stage, self.mode, self.S)
            except Exception as e:
                raise Fatal(f"cannot start the live desktop: {e}") from e
            if self._resume is not None:
                path, start = self._resume
                self._resume = None
                self._start_app(path, start)
        elif space:
            self._stop_picom()
            try:
                self._lead = self.S.space_predict / 1000.0
                path = self._control().track(self._lead)
                self.stage = presenter.Stage(self.mode.width, self.mode.height)
                self.space = Space(self.stage, self.mode, self.S, path, six_dof)
            except Exception as e:
                raise Fatal(f"cannot place the desktop in space: {e}") from e
            log(f"space: {self.S.space_mode}, {'6DOF' if six_dof else '3DOF'}")
            self._sync_grabs()
            if self.controls.refused:
                log(f"{self.controls.refused}: recenter from the panel instead")
        else:
            self._start_picom()

    def _leave_view(self) -> None:
        if self.app is not None:
            self._close_app()
        if self.desktop is not None:
            self.desktop.close()
            self.desktop = None
        if self.space is not None:
            self.space.close()
            self.space = None
            self._sync_grabs()
            try:
                self._control().untrack()
            except GlassesError as e:
                log(f"could not stop the head tracking: {e}")
        if self.stage is not None:
            self.stage.close()
            self.stage = None
        if self.panel_window is not None:
            self.panel_window.close()
            self.panel_window = None

    def _stop_picom(self) -> None:
        r = subprocess.run(["pgrep", "-x", "picom"], capture_output=True, text=True)
        pids = r.stdout.split()
        if not pids:
            return
        with open(f"/proc/{pids[0]}/cmdline", "rb") as f:
            self._picom = [a.decode() for a in f.read().split(b"\0") if a]
        subprocess.run(["pkill", "-x", "picom"])
        deadline = time.monotonic() + 3.0
        while subprocess.run(["pgrep", "-x", "picom"], capture_output=True).returncode == 0:
            if time.monotonic() > deadline:
                raise Fatal("picom did not exit")
            time.sleep(0.05)

    def _start_picom(self) -> None:
        if self._picom:
            subprocess.Popen(self._picom, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True)
            self._picom = None

    # --- modes -----------------------------------------------------------------

    def _control(self) -> Control:
        if self.control is None:
            raise SessionError("the glasses' control channel is not open")
        return self.control

    def _open_control(self, six_dof: bool) -> Control:
        return Control(on_mode=lambda code: self.events.put(("glasses-mode", code)),
                       six_dof=six_dof)

    def _ensure_control(self, six_dof: bool) -> None:
        """The SDK picks 3DOF or 6DOF when it starts: reopen it to change."""
        if self._control().six_dof == six_dof:
            return
        log(f"glasses tracking: {'6DOF' if six_dof else '3DOF'}")
        self.control.close()
        self.control = None
        try:
            self.control = self._open_control(six_dof)
        except GlassesError as e:
            raise Fatal(f"the glasses' SDK did not come back: {e}") from e

    def _sync_view(self) -> None:
        """Follow the placement settings: in a 2D mode SPACE MODE and TRACKING
        pick the view; an open model follows its own TRACKING; both pass on a
        new PREDICT."""
        try:
            if isinstance(self.app, ModelView):
                six_dof = self.S.model_dof == "6dof"
                if self.app.six_dof != six_dof:
                    path = self.app.path
                    self._close_app()
                    self._start_app(path, 0.0)
                else:
                    self._retrack()
                return
            if self.mode.sbs or self.app is not None:
                return
            space, six_dof = self._space_wanted()
            if space != (self.space is not None) or (space and self.space.six_dof != six_dof):
                self._leave_view()
                self._enter_view()
            elif space:
                self._retrack()
        except (SessionError, GlassesError) as e:
            self._failed(e)

    def _retrack(self) -> None:
        if self.S.space_predict / 1000.0 != self._lead:
            self._lead = self.S.space_predict / 1000.0
            self._control().track(self._lead)

    def _failed(self, e: Exception) -> None:
        """Outside _drain: log a failure; end the session if it is fatal."""
        log(e)
        if isinstance(e, Fatal) or not self._glasses_connected():
            self.running = False

    def _light(self, mode: Mode, timing: displays.Timing) -> None:
        """Light the glasses alone in `mode`, behind a freshly published monitor."""
        displays.publish(self.output, mode.desktop)
        displays.light_alone(self.output, timing, displays.query())
        if not displays.wait_for(displays.scanning(self.output, timing), SCAN_TIMEOUT):
            raise Fatal(f"X did not light {self.output} at {timing}")
        self.mode = self.pending = mode
        time.sleep(SETTLE)

    def _offered(self, mode: Mode):
        """A wait_for predicate: the glasses' EDID prefers `mode`; returns its timing."""
        def check(outs):
            o = outs.get(self.output)
            if o and o.connected and o.preferred and modes.by_timing(*o.preferred) == mode:
                return o.preferred
            return None
        return check

    def _retune(self, mode: Mode, send: bool) -> None:
        """Take the glasses to `mode` and drive it: ask them (send), or follow a
        change they made themselves. The view must already be down."""
        displays.publish(self.output, mode.desktop)
        if send:
            # X is driving the glasses at their current mode, the one state in
            # which they take a change.
            try:
                self._control().set_mode(mode)
            except GlassesError:
                displays.publish(self.output, self.mode.desktop)
                raise
        timing = displays.wait_for(self._offered(mode), EDID_TIMEOUT)
        if timing is None:
            raise Fatal(f"the glasses did not come back in {mode.name}")
        self._light(mode, timing)

    def _change(self, mode: Mode, send: bool) -> None:
        log(f"{self.mode.name} -> {mode.name}{'' if send else ' (glasses button)'}")
        if self.app is not None:
            if mode.sbs:                     # another 3D mode: pick the app up there
                self._resume = (self.app.path, self.app.position())
            else:                            # to 2D: the app is over
                self._resume = self._app_return = None
        self._leave_view()
        try:
            self._retune(mode, send)
        except GlassesError:
            self._enter_view()               # still in the old mode: show it again
            raise
        self._enter_view()

    def _switch(self, mode: Mode) -> None:
        if mode != self.mode:
            self._change(mode, send=True)

    def _glasses_changed(self, code: int) -> None:
        """The glasses report a mode: an echo of our own switch, or their button."""
        if code == self.mode.code:
            return
        mode = modes.by_code(code)
        if mode is None:
            raise Fatal(f"the glasses switched to mode 0x{code:02x}, which braindance "
                        "does not drive")
        self._change(mode, send=False)

    # --- apps: a video, a 3D model --------------------------------------------------

    def _open(self, path: str, cmd: str) -> None:
        """`cmd` "play": play a video; "view": view a model or splat scene; on
        the stage in a 3D mode. The command says which, so a file one doesn't
        take is refused, never opened in the other app."""
        path = os.path.abspath(os.path.expanduser(path))
        if not os.path.isfile(path):
            raise SessionError(f"no such file: {path}")
        name = os.path.basename(path)
        if cmd == "view" and not is_model(path):
            raise SessionError(f"{name}: view takes a model or splats ({', '.join(MODEL_TYPES)})")
        if cmd == "play" and is_model(path):
            raise SessionError(f"{name} is a 3D model: braindance view shows it")
        if self.mode.sbs:
            if self.app is not None:
                self._close_app()
            self._start_app(path, 0.0)
            return
        # From 2D: switch to a 3D mode, the one nearest this one for a video, the
        # fastest for a model; the view starts the app once it is up, and its end
        # brings the glasses back.
        sbs = [m for m in modes.MODES if m.sbs]
        if is_model(path):
            target = max(sbs, key=lambda m: (m.hz, m.height))
        else:
            target = (next((m for m in sbs if (m.height, m.hz) == (self.mode.height, self.mode.hz)),
                           None)
                      or next((m for m in sbs if m.height == self.mode.height), None) or sbs[0])
        self._app_return = self.mode
        self._resume = (path, 0.0)
        self._switch(target)

    def _start_app(self, path: str, start: float) -> None:
        if is_model(path):
            six_dof = self.S.model_dof == "6dof"
            self._ensure_control(six_dof)
            self._lead = self.S.space_predict / 1000.0
            pose = self._control().track(self._lead)
            self.app = ModelView(self.stage, self.mode, path, self.S, pose, six_dof)
            log(f"viewing {path}, {self.S.model_mode}, {'6DOF' if six_dof else '3DOF'}")
        else:
            self.app = Player(self.stage, self.mode, path, self.S, start)
            log(f"playing {path}" + (f" from {start:.0f} s" if start > 1 else ""))
        self._sync_grabs()

    def _close_app(self) -> None:
        model = isinstance(self.app, ModelView)
        self.app.close()
        self.app = None
        self._sync_grabs()
        if model:
            try:
                self._control().untrack()
            except GlassesError as e:
                log(f"could not stop the head tracking: {e}")

    def _end_app(self) -> None:
        """The video ended or was stopped, or the model closed: back to the
        desktop (its SDK session back to 3DOF), or the 2D mode the app took us
        from."""
        if self.app.error:
            log(f"cannot open {self.app.path}: {self.app.error}")
        self._close_app()
        if self._app_return is not None:
            self.events.put(("switch", self._app_return))
            self._app_return = None
            return
        try:
            self._ensure_control(self._space_wanted()[1])
        except (SessionError, GlassesError) as e:
            self._failed(e)

    def _glasses_connected(self) -> bool:
        o = displays.query().get(self.output)
        return bool(o and o.connected) and present()

    # --- the loop ------------------------------------------------------------

    def _request(self, req: dict) -> dict:
        cmd = req.get("cmd")
        if cmd == "status":
            return {"ok": True, "output": self.output, "mode": self.mode.name,
                    "live3d": self.S.live3d, "glasses": self._control().info,
                    "playing": self.app.path if isinstance(self.app, Player) else None,
                    "viewing": self.app.path if isinstance(self.app, ModelView) else None,
                    "fps": round(self.desktop.fps, 1) if self.desktop and not self.app
                           else round(self.space.fps, 1) if self.space else None,
                    "depth": self.desktop.depth_state if self.desktop and not self.app else None,
                    "space": f"{self.S.space_mode}, {self.space.state}" if self.space else None}
        if cmd in ("play", "view"):
            self._open(req["path"], cmd)
            return {"ok": True, "mode": self.mode.name}
        if cmd == "mode":
            try:
                mode = modes.by_name(req["mode"])
            except KeyError as e:
                return {"ok": False, "error": str(e.args[0])}
            self._switch(mode)
            return {"ok": True, "mode": self.mode.name}
        if cmd == "quit":
            self.running = False
            return {"ok": True}
        return {"ok": False, "error": f"unknown command {cmd!r}"}

    def _drain(self) -> None:
        """Handle every queued event. A fatal error ends the session."""
        while self.running:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                return
            kind = event[0]
            try:
                if kind == "quit":
                    self.running = False
                elif kind == "ipc":
                    event[2](self._request(event[1]))
                elif kind == "glasses-mode":
                    self._glasses_changed(event[1])
                elif kind == "switch":
                    self._switch(event[1])
            except (SessionError, GlassesError) as e:
                log(e)
                if kind == "ipc":
                    event[2]({"ok": False, "error": str(e)})
                if isinstance(e, Fatal) or not self._glasses_connected():
                    self.running = False

    # --- input -------------------------------------------------------------------

    def _sync_grabs(self) -> None:
        """The pointer is ours while the panel is open or an app runs; the
        keyboard only while an app runs; the recenter button while the desktop
        or a model is placed in space."""
        self.controls.want(pointer=self.panel_open or self.app is not None,
                           keyboard=self.app is not None,
                           recenter=self.space is not None or isinstance(self.app, ModelView))

    def _toggle_panel(self) -> None:
        self.panel_open = not self.panel_open
        self._drag = self._panel_grab = self._panel_moving = None
        if self.panel_open:
            self.panel_opened_at = time.monotonic()
            self.pending = self.mode
            self.pointer = self.controls.pointer()
        self._sync_grabs()

    def _panel_xy(self) -> tuple[int, int]:
        """The panel's top-left on the desktop (root pixels)."""
        if self._panel_moving is not None:
            return self._panel_moving
        return panel.place(self.mode.desktop, self.S.panel_pos)

    def _panel_hit(self, x, y):
        """What the pointer at root (x, y) is on: see panel.hit."""
        if self.desktop is not None:
            p = self.desktop.renderer.panel
        elif self.space is not None:
            p = self.space.panel
        elif self.panel_window is not None:
            p = self.panel_window.panel
        else:
            return None
        ox, oy = self._panel_xy()
        return panel.hit(p.hits, x - ox, y - oy)

    def _panel_mouse(self, ev) -> None:
        """Over the panel: hover selects a row, the wheel steps it, a left click
        toggles / cycles / applies it, a right click steps it back; on the tab
        strip a click or the wheel switches tabs; a left drag anywhere else on it
        moves the panel, and its place is kept. Beside the panel, on the live 3D
        desktop: the wheel zooms, a left drag pans, a middle click resets; on the
        desktop placed in space: the wheel sizes it, a left drag moves an
        anchored screen, a middle click recenters it."""
        m = self.menu
        scene = (self.desktop or self.space) if self.app is None else None
        view = self.mode.desktop
        if ev.motion is not None:
            x, y = ev.motion
            if self._panel_grab is not None:
                gx, gy = self._panel_grab
                self._panel_moving = panel.place(view, panel.where(view, (x - gx, y - gy)))
            elif self._drag is not None and scene is not None:
                scene.pan(x - self._drag[0], y - self._drag[1])
                self._drag = (x, y)
            self.pointer = (x, y)
            h = self._panel_hit(x, y)
            if h and h[0] == "row":
                m.hover(h[1])
        for button, x, y, _ in ev.presses:
            self.pointer = (x, y)
            h = self._panel_hit(x, y)
            wheel = {4: +1, 5: -1}.get(button)
            if h is None:
                if scene is None:
                    continue
                if wheel:
                    scene.zoom(wheel)
                elif button == 1:
                    self._drag = (x, y)
                elif button == 2:
                    scene.reset_view()
            elif h[0] == "tab":
                if button == 1:
                    m.select_tab(h[1])
                elif wheel:
                    (m.prev_tab if wheel > 0 else m.next_tab)()
            elif h[0] == "row":
                if button == 1:
                    m.click(h[1])
                elif button == 3:
                    m.step(h[1], -1)
                elif wheel:
                    m.step(h[1], wheel)
            elif button == 1:                  # the panel itself: pick it up
                ox, oy = self._panel_xy()
                self._panel_grab = (x - ox, y - oy)
                self._panel_moving = (ox, oy)
        if any(b == 1 for b, _, _ in ev.releases):
            if self._panel_moving is not None:   # put it down: keep the place
                self.S.panel_pos = panel.where(view, self._panel_moving)
            self._drag = self._panel_grab = self._panel_moving = None

    def _input(self) -> None:
        ev = self.controls.update()
        if ev.panel:
            self._toggle_panel()
        if ev.recenter:
            self._recenter()
        if self.app is not None:             # the keyboard is the app's
            for name, shift in ev.keys:
                self.app.key(name, shift)
        if self.panel_open:
            self._panel_mouse(ev)
        elif self.app is not None:
            for button, x, y, shift in ev.presses:
                self.app.mouse("press", x, y, button, shift)
            for button, x, y in ev.releases:
                self.app.mouse("release", x, y, button)
            if ev.motion is not None:
                self.app.mouse("move", *ev.motion)

    def _persist(self) -> None:
        sig = settings.signature(self.S)
        if sig != self._saved:
            self._saved = sig
            settings.save(self.S)

    def _draw_app(self) -> None:
        """One frame of the app, with the panel over it when it is open: at the
        video's own plane, or the screen's for a model."""
        stage, now = self.stage, time.monotonic()
        stage.ctx.screen.use()
        alive = self.app.draw(ui=not self.panel_open)
        if self.panel_open:
            r = self.desktop.renderer
            names, ti = self.menu.tab_bar()
            r.panel.update(names, ti, self.menu.rows())
            r.draw_panel(self._panel_xy(), now - self._t0,
                         min(1.0, (now - self.panel_opened_at) / PANEL_APPEAR),
                         self.S.media_depth / 2 if isinstance(self.app, Player) else 0.0,
                         self.pointer)
        stage.swap()
        self.app.report_swap()
        stage.poll()
        if now - self._awake_at > AWAKE:
            self._awake_at = now
            self.controls.keep_awake()
        if not alive:
            self._end_app()

    def _draw(self) -> None:
        """One frame: an app or the live desktop in 3D, the desktop placed in
        space in 2D, else the 2D panel window while the panel is open. All block
        on vsync; with nothing to draw, idle."""
        if self.app is not None:
            self._draw_app()
            return
        view = self.desktop or self.space
        if view is not None:
            view.frame(self.menu if self.panel_open else None, self.panel_opened_at,
                       self.pointer, self._panel_xy())
            return
        if not self.panel_open:
            if self.panel_window is not None and self.panel_window.shown:
                self.panel_window.hide()
            time.sleep(IDLE)
            return
        if self.panel_window is None:
            self.panel_window = presenter.PanelWindow()
        if self.panel_window.shown:
            self.panel_window.move(self._panel_xy())
        else:
            self.panel_window.show(self._panel_xy())
        names, ti = self.menu.tab_bar()
        self.panel_window.panel.update(names, ti, self.menu.rows())
        now = time.monotonic()
        self.panel_window.draw(now - self._t0,
                               min(1.0, (now - self.panel_opened_at) / PANEL_APPEAR))

    def run(self, mode: Mode | None = None, app: tuple[str, str] | None = None) -> None:
        """Run the session; `app` ("play" or "view", path) opens a file once it is up."""
        server = ipc.Server(lambda req, reply: self.events.put(("ipc", req, reply)))
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, lambda *_: self.events.put(("quit",)))
        try:
            self.start()
            if mode is not None:
                self._switch(mode)
            if app is not None:
                self._open(app[1], app[0])
            checked = time.monotonic()
            while self.running:
                self._drain()
                if not self.running:
                    break
                now = time.monotonic()
                if now - checked >= 1.0:
                    checked = now
                    if not present():
                        log("glasses unplugged")
                        break
                self._input()
                self._persist()
                self._sync_view()
                if self.running:
                    self._draw()
        finally:
            server.close()
            self._persist()
            if self.snapshot:
                self.stop()
