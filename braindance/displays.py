"""displays.py — the X outputs: find the glasses, light them alone, put things back.

Everything goes through xrandr. For a whole session braindance publishes a
RandR monitor named "braindance" over the desktop it shows. That is its
contract with the rest of the desktop: qtile lays windows out inside it, and
indigoshell leaves the outputs alone while it exists. The monitor is always
published before the outputs change, because both react to the RandR event
the change raises, and not to the monitor itself.
"""
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field

MONITOR = "braindance"
GLASSES_EDID_NAME = "VITURE"
_MM_PER_PX = 25.4 / 96          # the published monitor claims a nominal 96 dpi

Timing = tuple[int, int, float]  # width, height, refresh


@dataclass
class Output:
    name: str
    connected: bool
    primary: bool = False
    rect: tuple[int, int, int, int] | None = None    # x, y, w, h while active
    current: Timing | None = None                    # being scanned out
    preferred: Timing | None = None                  # the sink's preferred (EDID) timing
    timings: list[Timing] = field(default_factory=list)
    edid_name: str | None = None

    @property
    def active(self) -> bool:
        return self.rect is not None


_HEADER = re.compile(r"^(\S+) (connected|disconnected)( primary)?(?: (\d+)x(\d+)\+(\d+)\+(\d+))?")
_TIMING = re.compile(r"^\s+(\d+)x(\d+)i?\s+(.*)$")
_RATE = re.compile(r"(\d+\.\d+)([* ]?)(\+?)")       # 60.00*+ = current, preferred
_HEX = re.compile(r"^\t\t([0-9a-f]+)$")


def _xrandr(*args: str, check: bool = True) -> str:
    return subprocess.run(["xrandr", *args], capture_output=True, text=True, check=check).stdout


def _edid_name(edid: bytes) -> str | None:
    """The monitor name descriptor (tag 0xFC) of an EDID base block."""
    for off in range(54, 126, 18):
        d = edid[off:off + 18]
        if len(d) == 18 and d[:3] == b"\0\0\0" and d[3] == 0xFC:
            return d[5:].split(b"\n")[0].decode(errors="replace").strip()
    return None


def query() -> dict[str, Output]:
    """Every output, from one `xrandr --props`."""
    outs: dict[str, Output] = {}
    cur: Output | None = None
    edid: list[str] | None = None
    for line in _xrandr("--props").splitlines():
        if edid is not None and cur is not None:
            h = _HEX.match(line)
            if h:
                edid.append(h.group(1))
                continue
            cur.edid_name = _edid_name(bytes.fromhex("".join(edid)))
            edid = None
        m = _HEADER.match(line)
        if m:
            name, state, primary = m.group(1), m.group(2), m.group(3)
            rect = ((int(m.group(6)), int(m.group(7)), int(m.group(4)), int(m.group(5)))
                    if m.group(4) else None)
            cur = outs[name] = Output(name, state == "connected", bool(primary), rect)
            continue
        if cur is None:
            continue
        if line == "\tEDID: ":
            edid = []
            continue
        t = _TIMING.match(line)
        if t and not line.startswith("\t"):
            w, h = int(t.group(1)), int(t.group(2))
            for rate, star, plus in _RATE.findall(t.group(3)):
                timing = (w, h, float(rate))
                cur.timings.append(timing)
                if star == "*":
                    cur.current = timing
                if plus:
                    cur.preferred = timing
    return outs


def glasses(outs: dict[str, Output]) -> Output | None:
    """The connected output whose EDID names VITURE."""
    return next((o for o in outs.values()
                 if o.connected and o.edid_name and GLASSES_EDID_NAME in o.edid_name), None)


def wait_for(predicate, timeout: float, interval: float = 0.1):
    """Poll query() until predicate(outs) is truthy; return that value, or None."""
    deadline = time.monotonic() + timeout
    while True:
        result = predicate(query())
        if result or time.monotonic() >= deadline:
            return result or None
        time.sleep(interval)


# --- the published desktop ---------------------------------------------------

def publish(output: str, size: tuple[int, int]) -> None:
    """(Re)publish the braindance monitor: `size` at the origin, on `output`."""
    w, h = size
    _xrandr("--delmonitor", MONITOR, check=False)
    _xrandr("--setmonitor", MONITOR,
            f"{w}/{round(w * _MM_PER_PX)}x{h}/{round(h * _MM_PER_PX)}+0+0", output)


def withdraw() -> None:
    _xrandr("--delmonitor", MONITOR, check=False)


def published() -> bool:
    return any(line.split()[1].lstrip("+*") == MONITOR
               for line in _xrandr("--listmonitors").splitlines()[1:])


# --- lighting outputs --------------------------------------------------------

def _rate(r: float) -> str:
    return f"{r:.2f}"


def _apply(args: list[str]) -> None:
    """One xrandr call for a whole layout: split into an off call and an on
    call, there is a moment with no output lit and X collapses the screen to
    640x480. NVIDIA intermittently fails a CRTC reassignment ("Configure crtc
    N failed"), so a failed call is tried once more after the CRTCs settle."""
    try:
        _xrandr(*args)
    except subprocess.CalledProcessError:
        time.sleep(0.5)
        _xrandr(*args)


def _offs(outs: dict[str, Output], keep: set[str]) -> list[str]:
    return sum((["--output", o.name, "--off"] for o in outs.values()
                if o.name not in keep and (o.active or o.connected)), [])


def light_alone(output: str, timing: Timing, outs: dict[str, Output]) -> None:
    """Make `output` the only active output, primary at the origin, at `timing`."""
    w, h, r = timing
    _apply(["--output", output, "--mode", f"{w}x{h}", "--rate", _rate(r),
            "--pos", "0x0", "--primary", *_offs(outs, {output})])


def scanning(output: str, timing: Timing):
    """A wait_for predicate: `output` is active and scanning out `timing`."""
    def check(outs):
        o = outs.get(output)
        return bool(o and o.active and o.current and o.current[:2] == timing[:2]
                    and abs(o.current[2] - timing[2]) < 1.5)
    return check


# --- the layout a session found, and putting it back -------------------------

def snapshot(outs: dict[str, Output]) -> list[dict]:
    return [{"name": o.name, "timing": o.current, "pos": o.rect[:2], "primary": o.primary}
            for o in outs.values() if o.active and o.current]


def restore(snap: list[dict]) -> None:
    """Put the snapshot's outputs back as they were, and turn every other off."""
    outs = query()
    args: list[str] = []
    for s in snap:
        if s["name"] not in outs or not outs[s["name"]].connected:
            continue
        w, h, r = s["timing"]
        args += ["--output", s["name"], "--mode", f"{w}x{h}", "--rate", _rate(r),
                 "--pos", f"{s['pos'][0]}x{s['pos'][1]}"]
        if s["primary"]:
            args.append("--primary")
    if args:
        _apply(args + _offs(outs, {s["name"] for s in snap}))


def snapshot_path() -> str:
    return os.path.join(os.environ.get("XDG_RUNTIME_DIR", "/tmp"), "braindance", "layout.json")


def save_snapshot(snap: list[dict]) -> None:
    os.makedirs(os.path.dirname(snapshot_path()), exist_ok=True)
    with open(snapshot_path(), "w") as f:
        json.dump(snap, f)


def load_snapshot() -> list[dict] | None:
    try:
        with open(snapshot_path()) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def forget_snapshot() -> None:
    try:
        os.remove(snapshot_path())
    except FileNotFoundError:
        pass
