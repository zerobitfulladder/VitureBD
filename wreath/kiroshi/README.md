# kiroshi

An earlier experiment: a head-tracked virtual desktop on VITURE Luma Ultra
glasses, with the Linux desktop shown as a screen locked in place in 6DoF. It
was built as an extension of the glasses demo that ships with VITURE's XR
Glasses SDK v2.

Only my own files are in this repository:

| File | What it does |
|---|---|
| `window_capture.cpp`, `window_capture.h` | A small GPU compositor: redirects every top-level X11 window (XComposite), binds each one's pixmap as a GL texture (GLX texture_from_pixmap) and reports its rectangle, so the renderer can lay the windows out as one desktop. |
| `scene_window.h` | The scene that draws those captured windows as a world-locked screen in front of you. |

The demo they plug into, and the SDK it builds against, are copyright VITURE
Inc. and are not included. To build kiroshi, get the SDK from VITURE, then add
these files to its demo and register the window scene beside the demo's own
scenes.

The capture has one hard requirement: no other compositor (picom) may redirect
the same windows, because the NVIDIA driver refuses a second GLX pixmap.

[`../braindance`](../braindance) grew out of this capture code and does not
need the SDK at all.
