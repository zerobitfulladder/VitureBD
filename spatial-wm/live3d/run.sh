#!/usr/bin/env bash
# Run live3d. GPU window capture needs picom OFF (NVIDIA tfp collision); stop it
# and restore on exit. Uses the nunif venv (torch + transformers + glfw/moderngl
# + python-xlib). Put the glasses in 3D mode (3840x1080) first.
#
# By default the desktop is moved onto the eDP head (the laptop panel, driven by
# the NVIDIA GPU in EnvyControl nvidia-mode) and live3d grabs it from there, so
# the rendered desktop and the glasses overlay live on different physical
# outputs -- no fullscreen stacking race. On exit the eDP head is turned back
# off and qtile is reloaded, restoring the glasses-only layout.
#   ./run.sh                 # FLAT to start; side button toggles to DEPTH
#   ./run.sh --mode depth    # start in DEPTH
#   ./run.sh --no-edp        # legacy: capture the glasses' own left half
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
py="$here/../../nunif/.venv/bin/python"
export DISPLAY="${DISPLAY:-:0}"

# No compositor runs while we capture (picom is stopped below), so force the
# NVIDIA driver to sync buffer swaps to vblank — reduces tearing.
export __GL_SYNC_TO_VBLANK=1

DESK="eDP-1-1"          # NVIDIA-driven laptop panel used as the desktop head
GLASSES="DP-0"          # 3840x1080 SBS glasses output

# --no-edp opts out of the separate-head model (legacy left-half capture).
use_edp=1
argv=()
for a in "$@"; do
    if [ "$a" = "--no-edp" ]; then use_edp=0; else argv+=("$a"); fi
done

qtile_reload() { qtile cmd-obj -o cmd -f reload_config >/dev/null 2>&1 || true; }

edp_off=0
restore() {
    pgrep -x picom >/dev/null || { echo "[run] restarting picom"; nohup $picom_cmd >/dev/null 2>&1 & }
    if [ "$edp_off" = 1 ]; then
        echo "[run] disabling $DESK, restoring glasses layout"
        xrandr --output "$DESK" --off || true
        qtile_reload
    fi
}
trap restore EXIT

region_arg=()
if [ "$use_edp" = 1 ]; then
    echo "[run] enabling $DESK as the desktop head (right of $GLASSES)"
    xrandr --output "$DESK" --mode 1920x1080 --right-of "$GLASSES"
    edp_off=1
    sleep 0.3
    qtile_reload                       # re-runs detect_screens -> desktop on eDP
    sleep 0.3
    geom="$(xrandr --current | grep -oP "^${DESK} connected.*?\K[0-9]+x[0-9]+\+[0-9]+\+[0-9]+" | head -1)"
    if [ -z "$geom" ]; then echo "[run] ERROR: $DESK did not come up" >&2; exit 1; fi
    # WxH+X+Y  ->  X,Y,W,H
    region="$(echo "$geom" | sed -E 's/([0-9]+)x([0-9]+)\+([0-9]+)\+([0-9]+)/\3,\4,\1,\2/')"
    echo "[run] capture region $region (from $DESK $geom)"
    region_arg=(--region "$region")
fi

picom_cmd="picom -b --config $HOME/.config/picom/picom.conf"
running="$(pgrep -ax picom | head -1 | cut -d' ' -f2- || true)"
[ -n "$running" ] && picom_cmd="$running"

if pgrep -x picom >/dev/null; then echo "[run] stopping picom for GPU capture"; pkill -x picom || true; sleep 0.4; fi

cd "$here"
"$py" -m live3d "${region_arg[@]}" "${argv[@]}"
