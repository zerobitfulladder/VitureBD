#!/usr/bin/env bash
# Run stereo-mirror. GPU window capture (texture_from_pixmap) collides with picom
# on NVIDIA, so picom is stopped for the duration and restarted on exit.
#
# Put the glasses in 3D mode FIRST (hardware button / official app) so the panel
# is 3840x1080, then launch this. Confine the desktop to 1920x1080 with qtile's
# fake_screens (see README.md).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"

# Command to restart picom on exit. Prefer the currently-running invocation, but
# fall back to the standard config-based one so a prior crashed run (which left
# picom off) doesn't mean picom never comes back.
picom_fallback="picom -b --config $HOME/.config/picom/picom.conf"
picom_cmd="$(pgrep -ax picom | head -1 | cut -d' ' -f2- || true)"
[ -z "${picom_cmd:-}" ] && picom_cmd="$picom_fallback"

restore_picom() {
    if ! pgrep -x picom >/dev/null; then
        echo "[run] restarting picom"
        # shellcheck disable=SC2086
        nohup $picom_cmd >/dev/null 2>&1 &
    fi
}
trap restore_picom EXIT

if pgrep -x picom >/dev/null; then
    echo "[run] stopping picom for GPU capture"
    pkill -x picom || true
    sleep 0.4
fi

# Not exec'd: regain control afterwards so the EXIT trap restarts picom.
"$here/build/stereo-mirror" "$@"
