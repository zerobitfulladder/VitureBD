# live3d — live desktop → stereo 3D

Captures the desktop, estimates monocular depth (Depth Anything V2 Small),
mesh-warps it to a stereo pair, and presents SBS fullscreen on the glasses — all
live. No feedback loop (window-pixmap capture, not screen-grab). Replaces the
SBS-passthrough stereo-mirror.

## Display heads

By default the desktop is rendered on a **separate head** — the eDP laptop panel,
which the NVIDIA GPU drives directly in EnvyControl nvidia-mode — and live3d
grabs windows from there while its overlay owns the glasses (DP-0). Because the
rendered desktop and the overlay are on different physical outputs, a window
going fullscreen can't restack above the overlay: **no stacking race.** `run.sh`
enables the eDP head, moves qtile's desktop onto it (`detect_screens`), and turns
it back off on exit. `--no-edp` falls back to the legacy single-display mode
(capture the glasses' own left 1920×1080), where the overlay keeps itself raised
above fullscreen windows.

## Architecture (hybrid: C++ capture + Python app)

```
capture/   libdesktopcapture.so   window-pixmap capture into the shared GL context (C++)
live3d/
  overlay.py  glfw override-redirect 3840x1080 window + moderngl context
  capture.py  ctypes bridge; wraps captured GL textures as moderngl externals
  depth.py    pluggable depth backends (DA2 torch-fp16 now; TensorRT slot later)
  render.py   composite windows -> work RT; FLAT / DEPTH (mesh-warp DIBR); zoom/pan/disparity; HUD
  input.py    Xlib Super+Alt key chords + mouse grabs
  app.py      orchestration + async depth thread (display 60fps, depth reused)
```

Display runs at vsync (60fps); in DEPTH mode the depth model runs on a worker
thread and the display reuses the latest map, so motion stays smooth even though
depth refreshes at ~30–40fps.

## Build & run

```bash
# one-time: build the capture lib
cd capture && cmake -S . -B build && cmake --build build -j && cd ..

# run (glasses in 3D mode first); uses the local .venv (torch + tensorrt)
./run.sh                  # starts FLAT; toggle to DEPTH with the side button
./run.sh --mode depth     # start in DEPTH
```

## Controls (hold **Super+Alt**)

| Input | Action |
|---|---|
| Super+Alt + **Q** | quit |
| Super+Alt + **scroll** | zoom in / out |
| Super+Alt + **left-drag** | pan |
| Super+Alt + **= / -** | depth strength (disparity) |
| Super+Alt + **side button 8** | reset zoom + pan |
| Super+Alt + **side button 9** | toggle FLAT ↔ DEPTH |

## Status / TODO

- First on-glasses run verifies the two untested keystones: the C++→moderngl GL
  texture sharing (capture bridge) and image orientation. Flip fixes expected,
  as with stereo-mirror.
- TensorRT backend (cu12-matched onnxruntime or torch-tensorrt) — slots into
  `depth.py` behind `make_depth`; torch-fp16 runs now.
- Cursor is not yet drawn (hardware cursor hidden TODO); GL→CUDA interop to drop
  the depth readback round-trip is a later optimization.
