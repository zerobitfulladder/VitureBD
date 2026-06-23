#!/usr/bin/env bash
# Run live3d. GPU window capture needs picom OFF (NVIDIA tfp collision); stop it
# and restore on exit. Uses the nunif venv (torch + transformers + glfw/moderngl
# + python-xlib). Put the glasses in 3D mode (3840x1080) first.
#   ./run.sh                 # FLAT to start; side button toggles to DEPTH
#   ./run.sh --mode depth    # start in DEPTH
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
py="$here/../../nunif/.venv/bin/python"

# No compositor runs while we capture (picom is stopped below), so force the
# NVIDIA driver to sync buffer swaps to vblank — reduces tearing.
export __GL_SYNC_TO_VBLANK=1

picom_cmd="picom -b --config $HOME/.config/picom/picom.conf"
running="$(pgrep -ax picom | head -1 | cut -d' ' -f2- || true)"
[ -n "$running" ] && picom_cmd="$running"
restore() { pgrep -x picom >/dev/null || { echo "[run] restarting picom"; nohup $picom_cmd >/dev/null 2>&1 & }; }
trap restore EXIT

if pgrep -x picom >/dev/null; then echo "[run] stopping picom for GPU capture"; pkill -x picom || true; sleep 0.4; fi

cd "$here"
"$py" -m live3d "$@"
