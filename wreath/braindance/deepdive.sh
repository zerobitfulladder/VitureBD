#!/usr/bin/env bash
# BRAINDANCE — thin entry: env + venv -> the Python session manager
# (braindance/launch.py). It detects outputs + glasses, shows the cyberpunk
# display-profile picker, switches displays (xrandr + qtile restart), stops picom
# for GPU capture, runs the depth pipeline, and restores the previous layout on
# exit. Put the glasses in 3D (SBS) mode first for the braindance profile.
#
#   ./deepdive.sh                                # GUI picker, then run
#   ./deepdive.sh --profile braindance           # skip the picker
#   ./deepdive.sh --profile monitor              # just switch to the 2K monitor
#   ./deepdive.sh --profile glasses-flat         # glasses as a normal 1080p display
#   ./deepdive.sh --dry-run                      # print xrandr, apply nothing
#   ./deepdive.sh --profile braindance -- --trt --mode depth   # forward to the app
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
export DISPLAY="${DISPLAY:-:0}"

# No compositor runs while we capture (launcher stops picom), so force the NVIDIA
# driver to sync buffer swaps to vblank — reduces tearing.
export __GL_SYNC_TO_VBLANK=1

cd "$here"
exec "$here/.venv/bin/python" -m braindance "$@"
