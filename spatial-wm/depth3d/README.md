# depth3d — 2D→3D via monocular depth (prototype)

Proof-of-concept to validate **quality / comfort / FPS** before integrating into
the C++ `stereo-mirror` app. Reads a **2D video file**, estimates depth with
Depth Anything V2 (Small) on the GPU, synthesizes a stereo pair with a GPU
mesh-warp (hole-free DIBR), and presents it **fullscreen side-by-side** to the
glasses (put them in 3D SBS mode first).

The video file is only the demo input — the real target is the live desktop
capture already done in `../stereo-mirror`. A frame is a frame; the depth + warp
don't care about the source, so this fully validates the approach.

## Setup (uv)

Already created: `.venv` (Python 3.12) with torch+cu124, transformers, opencv
(headless), glfw, moderngl. To recreate:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv/bin/python transformers opencv-python-headless glfw moderngl pillow
```

## Run

Glasses in 3D mode (3840×1080), then:

```bash
.venv/bin/python depth3d.py /path/to/any_2d_video.mp4
# better quality, ~35-40 fps:  --in-h 392   (default)
# faster, ~55-60 fps:          --in-h 294
```

Useful flags: `--disp 18` (max disparity px, 3D strength), `--conv 0.5`
(convergence / screen plane), `--smooth 0.6` (temporal stability), `--in-h`
(depth resolution / speed), `--panel 3840x1080`.

## Controls (focus the GL window)

| Key       | Action                          |
|-----------|---------------------------------|
| `-` `=`   | disparity (3D strength) − / +   |
| `[` `]`   | convergence (screen plane)*     |
| `,` `.`   | temporal smoothing − / +        |
| `←` `→`   | seek −10s / +10s                |
| `s`       | swap left/right eye             |
| `m`       | mono preview (depth off)        |
| `space`   | pause                           |
| `r`       | reset tuning                    |
| `q`/Esc   | quit                            |

\* convergence is a uniform per-eye shift; in fullscreen video the eyes adapt to
it so it looks subtle. Disparity is the main 3D control.

The console prints `display X | process Y fps (Z ms)`: **display** is vsync-capped;
**process** is the real capacity (how fast we *could* go) — if it's ≥60 we can hit
60fps. Each eye is 1920×1080, so download videos at **1080p** (4K just slows decode
with no per-eye benefit). Edges look sharper with `--model
depth-anything/Depth-Anything-V2-Base-hf` or a higher `--in-h` (both cost FPS).

## What to judge

- Does it fuse into comfortable depth? Tune `-`/`=` then `[`/`]`.
- Is the 3D convincing, or too "cardboard-cutout" / flickery? `,`/`.` for stability.
- Artifacts at object edges (mesh-warp stretches over occlusions — expected).
- FPS (printed each second). If good, we port depth + warp into stereo-mirror.
