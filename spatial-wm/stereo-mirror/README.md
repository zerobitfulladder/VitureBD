# stereo-mirror

Duplicate a 1920×1080 desktop into both eyes of the Viture glasses while they are
in **3D side-by-side mode** (3840×1080), and split stereo SBS video/images so each
eye sees its own half un-squished.

No device/IMU/SDK code — it only captures the desktop and draws it twice. Put the
glasses in 3D mode yourself (hardware button / official app) before launching.

## Why it works this way

In 3D mode the glasses demand a **3840×1080** signal and split it themselves:
left 1920 → left eye, right 1920 → right eye. So the X root is unavoidably 3840
wide. You don't shrink X; you confine the *usable desktop* to the left 1920×1080
and let this program paint over the whole panel, drawing that box twice.

```
  X root (= glasses panel): 3840 x 1080
  ┌───────────────────────┬───────────────────────┐
  │ qtile lives here       │   dead space (unused) │
  │ 1920 x 1080  (captured)│                       │
  └───────────────────────┴───────────────────────┘
        │ captured & drawn twice by this overlay
        ▼
  ┌───────────────────────┬───────────────────────┐
  │  LEFT eye              │  RIGHT eye             │
  │  FLAT: identical copy. SBS (F2): left/right     │
  │  half of the frame, each stretched to fill.     │
  └───────────────────────┴───────────────────────┘
```

## Confine qtile to 1920×1080

In `~/.config/qtile/config.py`:

```python
from libqtile.config import Screen
fake_screens = [
    Screen(x=0, y=0, width=1920, height=1080),   # add bars/widgets as you like
]
```

`fake_screens` makes qtile manage only that box — including maximize and
fullscreen — so a fullscreened browser video is 1920×1080, not 3840.

## Build & run

```bash
cmake -S . -B build && cmake --build build -j
./run.sh            # stops picom (NVIDIA tfp collision), restarts on exit
```

## Controls — all require the **Super+Alt** chord

Every control is read via `XQueryKeymap` and gated behind **Super+Alt**, so bare
keys never trigger the program and the combos don't disturb qtile (which never
binds Super+Alt) or focused apps (which don't use Super). Hold Super+Alt, then:

| Chord                    | Action                                       |
|--------------------------|----------------------------------------------|
| Super+Alt+**Q**          | quit                                         |
| Super+Alt+**S**          | toggle FLAT ↔ SBS                            |
| Super+Alt+**W**          | swap left/right halves (cross-eyed content)  |
| Super+Alt+**0**          | reset zoom / pan / convergence / align       |
| Super+Alt+**=** / **-**  | zoom in / out (both modes)                   |
| Super+Alt+**arrows**     | pan (both modes)                             |
| Super+Alt+**[** / **]**  | convergence − / + (horizontal disparity, SBS)|
| Super+Alt+**;** / **'**  | vertical alignment − / + (SBS)               |

### Mouse (also Super+Alt; grabbed so apps don't see them)

| Chord                              | Action                       |
|------------------------------------|------------------------------|
| Super+Alt + **scroll**             | zoom in / out                |
| Super+Alt + **middle-drag**        | pan                          |
| Super+Alt + **left-drag** (x)      | convergence adjust (SBS)     |
| Super+Alt + **side button (X9)**   | reset zoom + pan             |
| Super+Alt + **side button (X8)**   | toggle FLAT ↔ SBS            |

Side buttons default to X buttons 9 (reset) and 8 (toggle). If yours are
reversed or numbered differently, set `SWM_BTN_RESET=<n>` / `SWM_BTN_TOGGLE=<n>`
(find the numbers with `xev`). Bare clicks/scroll pass straight through to the
desktop — only the Super+Alt combos are intercepted.

The drawn cursor is scaled by 1/zoom so it stays a constant on-screen size.

Workflow: browse normally (FLAT, both eyes identical) → fullscreen an SBS video →
**Super+Alt+S** (or side button) → tune convergence with **Super+Alt+[** / **]**
until it fuses comfortably. The overlay keeps itself raised above fullscreen
video each frame.

## Env vars

- `SWM_REGION="WxH+X+Y"` — capture region (default: left half of the panel).
- `SWM_PANEL="WxH"` — override detected panel size (dev/testing).
- `SWM_NO_OVERLAY=1` — normal window instead of the fullscreen overlay (dev).
- `SWM_NO_HUD=1` — hide the on-screen status text.
- `SWM_CONV=<px>` — initial convergence value.
