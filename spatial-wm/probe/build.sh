#!/usr/bin/env bash
# Build the texture_from_pixmap capture probe.
set -euo pipefail
cd "$(dirname "$0")"
gcc -O2 -Wall -Wextra -o capture_probe capture_probe.c \
    $(pkg-config --cflags --libs x11 xcomposite gl) \
    -lGL
echo "built: $(pwd)/capture_probe"
