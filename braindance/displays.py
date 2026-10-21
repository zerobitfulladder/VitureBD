"""displays.py — detect outputs, switch display profiles, restore them.

Replaces the xrandr/qtile logic that used to live in run.sh. ONE tool arranges
the displays for every mode: the braindance 3D session, plain monitor use, or the
glasses as a normal flat display. Profiles live in ~/.config/braindance/profiles.json
(auto-created with sensible defaults) so you don't re-decide each time.

The model: braindance always needs exactly one "desktop head" (a real output it
captures) plus the glasses overlay. A profile just says which output is which
and what to turn off.
"""
import json
import os
import re
import subprocess
import time

VITURE_VID = "35ca"                 # Viture USB vendor id (see SDK device_enum.cpp)
# project-local config (under braindance/config/, not ~/.config)
CFG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config"))
PROFILES_PATH = os.path.join(CFG_DIR, "profiles.json")


# --- xrandr inspection -----------------------------------------------------
def _xrandr(*args):
    return subprocess.run(["xrandr", *args], capture_output=True, text=True).stdout


def outputs():
    """name -> {connected, active, primary, mode 'WxH', pos (x,y), modes:[...]}."""
    res, cur = {}, None
    for line in _xrandr("--query").splitlines():
        m = re.match(r"^(\S+) (connected|disconnected)(.*)$", line)
        if m:
            name, state, rest = m.groups()
            geo = re.search(r"\b(\d+)x(\d+)\+(\d+)\+(\d+)", rest)
            cur = {
                "connected": state == "connected",
                "primary": "primary" in rest,
                "active": bool(geo),
                "mode": f"{geo.group(1)}x{geo.group(2)}" if geo else None,
                "pos": (int(geo.group(3)), int(geo.group(4))) if geo else None,
                "modes": [],
            }
            res[name] = cur
        elif cur is not None:
            mm = re.match(r"^\s+(\d+x\d+)\s", line)
            if mm:
                cur["modes"].append(mm.group(1))
    return res


def _output_rect(name):
    o = outputs().get(name)
    if o and o["active"] and o["pos"] and o["mode"]:
        w, h = (int(v) for v in o["mode"].split("x"))
        return (o["pos"][0], o["pos"][1], w, h)
    return None


# --- glasses detection -----------------------------------------------------
def glasses_usb_present():
    """True if a Viture USB device (VID 0x35CA) is plugged in — works even before
    the glasses are switched to 3D mode (USB data is separate from DP video)."""
    base = "/sys/bus/usb/devices"
    try:
        for d in os.listdir(base):
            p = os.path.join(base, d, "idVendor")
            if os.path.isfile(p):
                with open(p) as f:
                    if f.read().strip().lower() == VITURE_VID:
                        return True
    except OSError:
        pass
    return False


def detect_glasses(outs, glasses_output="DP-0"):
    """Combine USB presence (plugged?) with xrandr (in 3D mode?). 3D = the glasses
    expose a 3840-wide mode and are driving it."""
    o = outs.get(glasses_output, {})
    width = int(o["mode"].split("x")[0]) if o.get("mode") else 0
    # Pick the SBS mode: width >= 3840. These are 1080p-per-eye glasses, so the
    # real mode is 3840x1080 -> prefer height 1080. Some panels also advertise a
    # spurious taller mode (e.g. 3840x1200) the CRTC can't drive ("Configure crtc
    # failed"); never pick it over 1080.
    mode3d, best = None, None
    for m in o.get("modes", []):
        w, h = (int(v) for v in m.split("x"))
        if w < 3840:
            continue
        score = (0 if h == 1080 else 1, h)   # 1080 first, else the smallest height
        if best is None or score < best:
            best, mode3d = score, m
    usb = glasses_usb_present()
    in_3d = bool(o.get("active") and width >= 3840)
    if not usb and not o.get("connected"):
        status = "not detected"
    elif in_3d:
        status = f"3D ready ({o['mode']})"
    elif o.get("connected"):
        status = "connected — not in 3D mode (switch to SBS)"
    else:
        status = "USB plugged — no video (switch to SBS)"
    return {
        "output": glasses_output, "usb": usb, "connected": o.get("connected", False),
        "active": o.get("active", False), "width": width,
        "in_3d": in_3d, "can_3d": mode3d is not None, "mode3d": mode3d,
        "status": status,
    }


# --- profiles --------------------------------------------------------------
def default_profiles():
    return {
        "braindance": {
            "label": "Braindance", "order": 0, "run": True,
            "desc": "Glasses (3D) + eDP head @1080p60, monitor off — depth pipeline.",
            "glasses": "DP-0", "head": "eDP-1-1",
            "head_mode": "1920x1080", "head_rate": "60", "off": ["HDMI-0"],
        },
        "monitor": {
            "label": "Monitor", "order": 1, "run": False,
            "desc": "Normal desktop on the 2K monitor; glasses + eDP off.",
            "primary": "HDMI-0", "primary_mode": None, "off": ["eDP-1-1", "DP-0"],
        },
        "glasses-flat": {
            "label": "Glasses (flat)", "order": 2, "run": False,
            "desc": "Glasses as a normal 1080p display; monitor + eDP off.",
            "primary": "DP-0", "primary_mode": "1920x1080", "off": ["HDMI-0", "eDP-1-1"],
        },
    }


def load_profiles():
    os.makedirs(CFG_DIR, exist_ok=True)
    if not os.path.exists(PROFILES_PATH):
        with open(PROFILES_PATH, "w") as f:
            json.dump(default_profiles(), f, indent=2)
        return default_profiles()
    try:
        with open(PROFILES_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default_profiles()


# --- applying / restoring --------------------------------------------------
def qtile_cmd(func):
    subprocess.run(["qtile", "cmd-obj", "-o", "cmd", "-f", func],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def qtile_reload():
    qtile_cmd("reload_config")


def qtile_restart():
    # A full restart rebuilds qtile's Screen objects from scratch — necessary when
    # the screen SHAPE changes (fake_screens <-> screens), which reload_config does
    # NOT fully apply (windows stay bound to stale/off-screen geometry). Safe here:
    # config.py respawns the bar in @hook.subscribe.startup, by design. Windows are
    # preserved across a restart.
    qtile_cmd("restart")


def _settle(on=(), off=(), timeout=4.0):
    """Block until xrandr reflects the target layout (on=active, off=inactive),
    so a qtile reload reads the FINAL geometry, not a transitional one. Returns
    True if it settled, False on timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        o = outputs()
        if all(o.get(n, {}).get("active") for n in on) and \
           all(not o.get(n, {}).get("active") for n in off):
            return True
        time.sleep(0.1)
    return False


def _reconfigure_qtile():
    """After the displays have settled, rebuild qtile's screens. We RESTART (not
    reload) because the screen shape changes between profiles (fake_screens for the
    confined head <-> a normal screen), and only a restart re-creates the Screen
    objects so windows land on the visible output."""
    qtile_restart()


def apply_profile(name, profiles, glasses, dry_run=False):
    """Switch to the named profile via one xrandr call + a qtile restart. For a
    `run` profile, returns (region, overlay_rect): the capture region (head rect)
    and the glasses output rect for the override-redirect overlay. Non-run -> (None, None)."""
    p = profiles[name]
    cmd = ["xrandr"]
    region = overlay_rect = None
    if p.get("run"):
        if not glasses.get("can_3d"):
            raise RuntimeError(
                f"glasses ({p['glasses']}) not in 3D mode — plug in and switch to "
                f"SBS/3D first ({glasses.get('status')})")
        g, mode3d = p["glasses"], glasses["mode3d"]
        gw, gh = (int(v) for v in mode3d.split("x"))
        hw, hh = (int(v) for v in p["head_mode"].split("x"))
        # Head at the ORIGIN (x=0), glasses to the RIGHT. Keeping the head's
        # coordinate space at x=0 (same as normal HDMI use) means managed-window
        # coords don't shift across the session<->restore transition, so windows
        # can't get stranded off-screen. Head is primary (the desktop lives there).
        cmd += ["--output", p["head"], "--mode", p["head_mode"],
                "--rate", p["head_rate"], "--pos", "0x0", "--primary"]
        cmd += ["--output", g, "--mode", mode3d, "--pos", f"{hw}x0"]
        for off in p.get("off", []):
            cmd += ["--output", off, "--off"]
        region = (0, 0, hw, hh)
        overlay_rect = (hw, 0, gw, gh)
    else:
        cmd += ["--output", p["primary"], "--primary", "--pos", "0x0"]
        cmd += (["--mode", p["primary_mode"]] if p.get("primary_mode") else ["--auto"])
        for off in p.get("off", []):
            cmd += ["--output", off, "--off"]

    if dry_run:
        print("DRY:", " ".join(cmd))
        return region, overlay_rect
    # Free the outputs we're disabling in their OWN call first. Reassigning a CRTC
    # while simultaneously freeing another in one atomic xrandr call intermittently
    # fails on NVIDIA ("Configure crtc N failed"); doing the --off first lets the
    # CRTCs settle so the enable step has free ones to grab.
    offs = p.get("off", [])
    if offs:
        off_cmd = ["xrandr"]
        for o in offs:
            off_cmd += ["--output", o, "--off"]
        subprocess.run(off_cmd)            # best-effort; not fatal if already off
        time.sleep(0.2)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:  # one retry after the CRTCs settle
        time.sleep(0.4)
        subprocess.run(cmd, check=True)
    if p.get("run"):
        _settle(on=[p["glasses"], p["head"]], off=p.get("off", []))
    else:
        _settle(on=[p["primary"]], off=p.get("off", []))
    _reconfigure_qtile()
    if region:                                   # re-derive the rects from reality
        region = _output_rect(p["head"]) or region
        overlay_rect = _output_rect(p["glasses"]) or overlay_rect
    return region, overlay_rect


def snapshot(outs):
    """Remember the currently-active outputs so we can restore them on exit."""
    return [{"name": n, "mode": o["mode"], "pos": o["pos"], "primary": o["primary"]}
            for n, o in outs.items() if o["active"]]


def restore(snap, all_names, force_off=(), dry_run=False):
    """Re-apply the pre-session layout. `force_off` outputs (e.g. the glasses) are
    turned OFF regardless of the snapshot: leaving the glasses active after a
    session re-triggers qtile's detect_screens (its 3840-wide '3D' heuristic
    misfires), confining the desktop to a bogus rect. We never force the head off
    — that may be the user's only desktop (e.g. eDP on the road)."""
    cmd = ["xrandr"]
    force_off = set(force_off)
    on = [s for s in snap if s["name"] not in force_off]
    on_names = {s["name"] for s in on}
    for s in on:
        cmd += ["--output", s["name"], "--mode", s["mode"],
                "--pos", f"{s['pos'][0]}x{s['pos'][1]}"]
        if s["primary"]:
            cmd += ["--primary"]
    off = [n for n in all_names if n not in on_names]
    for n in off:
        cmd += ["--output", n, "--off"]
    if dry_run:
        print("DRY restore:", " ".join(cmd))
        return
    subprocess.run(cmd, check=True)
    _settle(on=[s["name"] for s in on], off=off)
    _reconfigure_qtile()


# --- picom (GPU capture needs it OFF) --------------------------------------
def stop_picom():
    running = subprocess.run(["pgrep", "-x", "picom"]).returncode == 0
    if running:
        subprocess.run(["pkill", "-x", "picom"])
        time.sleep(0.4)
    return running


def start_picom():
    cfg = os.path.expanduser("~/.config/picom/picom.conf")
    args = ["picom", "-b"] + (["--config", cfg] if os.path.exists(cfg) else [])
    subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
