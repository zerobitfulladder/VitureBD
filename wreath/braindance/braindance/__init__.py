"""braindance — live desktop → stereo 3D for the Viture glasses (single display).

Captures the desktop (window-pixmap, via the C++ libdesktopcapture), estimates
monocular depth (Depth Anything V2), mesh-warps to a stereo pair, and presents
SBS fullscreen on the glasses. See README.md for architecture.
"""
