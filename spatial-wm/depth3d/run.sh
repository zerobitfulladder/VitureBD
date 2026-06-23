#!/usr/bin/env bash
# Always run depth3d with the nunif venv — it has everything for BOTH backends
# (Depth Anything via transformers AND iw3 Video Depth Anything + its deps).
#   ./run.sh ../../video/lucy.mp4              # per-frame Depth Anything
#   ./run.sh ../../video/lucy.mp4 --vda --in-h 294   # Video Depth Anything (stable)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
exec "$here/../../nunif/.venv/bin/python" "$here/depth3d.py" "$@"
