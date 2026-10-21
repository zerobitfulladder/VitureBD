"""cli.py — the braindance command.

  braindance [--mode NAME]   start a session on the glasses (refuses without them)
  braindance play FILE       play a 3D (side-by-side) video on the glasses: in the
                             running session, or in a new one
  braindance view FILE       view a 3D model (.glb, .gltf, .vrm) or Gaussian splats
                             (.splat, .ply) placed in the room, the same way
  braindance mode NAME       switch the running session's glasses to a mode
  braindance status          what the running session is driving
  braindance quit            end the running session and restore the displays
  braindance modes           the modes braindance drives the glasses in
  braindance restore         put the displays back after a session crashed
"""
import argparse
import os
import sys

from . import displays, ipc
from .glasses import modes


def _start(args, app=None) -> int:
    from .session import Session, SessionError
    mode = modes.by_name(args.mode) if args.mode else None
    try:
        Session().run(mode, app)
    except (SessionError, RuntimeError) as e:
        print(f"braindance: {e}", file=sys.stderr)
        return 1
    return 0


def _ask(req: dict) -> int:
    try:
        reply = ipc.request(req)
    except ipc.NotRunning as e:
        print(f"braindance: {e}", file=sys.stderr)
        return 1
    if not reply.get("ok"):
        print(f"braindance: {reply.get('error')}", file=sys.stderr)
        return 1
    if req["cmd"] == "status":
        g = reply["glasses"]
        print(f"{g['name']} (firmware {g['firmware']}) on {reply['output']}, {reply['mode']}")
        if reply.get("playing"):
            print(f"playing {reply['playing']}")
        elif reply.get("viewing"):
            print(f"viewing {reply['viewing']}")
        elif reply.get("space"):
            print(f"desktop in space ({reply['space']}), {reply['fps']:.0f} fps")
        elif reply.get("fps") is not None:
            print(f"live desktop, {'LIVE 3D' if reply['live3d'] else 'FLAT'}, {reply['fps']:.0f} fps, "
                  f"depth {reply['depth']}")
    elif req["cmd"] == "mode":
        print(reply["mode"])
    return 0


def _open(args) -> int:
    """play or view: in the running session, or in a new one."""
    from .viewer import MODEL_TYPES, is_model
    path = os.path.abspath(os.path.expanduser(args.file))
    if not os.path.isfile(path):
        print(f"braindance: no such file: {path}", file=sys.stderr)
        return 1
    if args.cmd == "view" and not is_model(path):
        print(f"braindance: view takes a model or splats ({', '.join(MODEL_TYPES)})",
              file=sys.stderr)
        return 1
    if args.cmd == "play" and is_model(path):
        print("braindance: that is a 3D model: braindance view shows it", file=sys.stderr)
        return 1
    if ipc.running():
        return _ask({"cmd": args.cmd, "path": path})
    return _start(args, app=(args.cmd, path))


def _modes(_args) -> int:
    for m in modes.MODES:
        print(f"{m.name:12s} {m.width}x{m.height} @ {m.hz} Hz"
              f"{'   side by side, desktop ' + 'x'.join(map(str, m.desktop)) if m.sbs else ''}")
    return 0


def _restore(_args) -> int:
    if ipc.running():
        print("braindance: a session is running; end it with `braindance quit`",
              file=sys.stderr)
        return 1
    snap = displays.load_snapshot()
    displays.withdraw()
    if snap is None:
        print("nothing to restore")
        return 0
    displays.restore(snap)
    displays.forget_snapshot()
    print("displays restored")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="braindance", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=[m.name for m in modes.MODES],
                    help="switch the glasses to this mode once the session is up")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("play", help="play a 3D video on the glasses")
    p.add_argument("file")
    p = sub.add_parser("view", help="view a 3D model or Gaussian splats placed in the room")
    p.add_argument("file")
    p = sub.add_parser("mode", help="switch the running session's glasses to a mode")
    p.add_argument("name", choices=[m.name for m in modes.MODES])
    sub.add_parser("status", help="what the running session is driving")
    sub.add_parser("quit", help="end the running session and restore the displays")
    sub.add_parser("modes", help="the modes braindance drives the glasses in")
    sub.add_parser("restore", help="put the displays back after a session crashed")
    args = ap.parse_args(argv)

    if args.cmd is None:
        return _start(args)
    if args.cmd in ("play", "view"):
        return _open(args)
    if args.cmd == "mode":
        return _ask({"cmd": "mode", "mode": args.name})
    if args.cmd in ("status", "quit"):
        return _ask({"cmd": args.cmd})
    return {"modes": _modes, "restore": _restore}[args.cmd](args)
