"""launch.py — the braindance session manager (entry point).

Flow:  detect outputs + glasses  ->  pick a display profile (cyberpunk GUI)  ->
apply it (xrandr + qtile)  ->  if it's the 3D profile, stop picom and run the app,
then restore the previous display layout on exit. Non-3D profiles just rearrange
the displays and exit (so you can use the glasses flat or go monitor-only).

  python -m braindance                       # GUI picker, then run
  python -m braindance --profile braindance      # skip the picker
  python -m braindance --profile monitor     # just switch displays, no app
  python -m braindance --dry-run             # print xrandr instead of running it
  python -m braindance --profile braindance -- --trt --mode depth   # forward args to app
"""
import argparse
import sys

from . import displays


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--profile", default=None, help="skip the picker; apply this profile")
    ap.add_argument("--no-picker", action="store_true",
                    help="auto-select (braindance if glasses are 3D-ready, else monitor)")
    ap.add_argument("--dry-run", action="store_true", help="print xrandr, don't apply")
    # everything after `--` (or any unknown args) is forwarded to the app
    args, app_args = ap.parse_known_args()
    app_args = [a for a in app_args if a != "--"]   # drop the passthrough separator

    outs = displays.outputs()
    glasses = displays.detect_glasses(outs)
    profiles = displays.load_profiles()
    snap = displays.snapshot(outs)

    name = args.profile
    if not name and args.no_picker:
        name = "braindance" if glasses.get("can_3d") and "braindance" in profiles else "monitor"
    if not name:
        from . import picker
        name = picker.choose(outs, glasses, profiles)
    if not name:
        print("[braindance] cancelled.")
        return
    if name not in profiles:
        sys.exit(f"[braindance] unknown profile '{name}'. have: {', '.join(profiles)}")

    try:
        region, overlay_rect = displays.apply_profile(
            name, profiles, glasses, dry_run=args.dry_run)
    except RuntimeError as e:
        sys.exit(f"[braindance] {e}")

    if not profiles[name].get("run"):
        print(f"[braindance] applied profile '{name}'.")
        return

    region_arg = []
    if region:
        region_arg += ["--region", ",".join(str(v) for v in region)]
    if overlay_rect:
        region_arg += ["--overlay-rect", ",".join(str(v) for v in overlay_rect)]
    if args.dry_run:
        print("DRY app:", "braindance", *region_arg, *app_args)
        return

    from . import app   # heavy (torch/glfw) — only when actually running
    picom_was = displays.stop_picom()
    try:
        app.main(region_arg + app_args)
    except KeyboardInterrupt:
        print("\n[braindance] interrupted — restoring displays.")
    finally:
        # restore the display layout FIRST (settle xrandr + restart qtile), THEN
        # bring picom back so it initializes against the final geometry. Force the
        # glasses output OFF so a still-plugged headset doesn't re-trigger qtile's
        # 3D-screen heuristic after we're done.
        glasses_out = profiles[name].get("glasses")
        displays.restore(snap, list(outs),
                         force_off=[glasses_out] if glasses_out else ())
        if picom_was:
            displays.start_picom()


if __name__ == "__main__":
    main()
