"""braindance — the desktop on VITURE glasses.

A session (session.py) owns the glasses and the X outputs: it lights the
glasses alone, switches their display modes through VITURE's SDK (glasses/),
and puts everything back when it ends or the glasses are unplugged. In 3D the
live desktop (desktop.py, capture.py, depth.py, stereo.py, render.py) captures
the desktop window by window, estimates depth, and warps a view for each eye;
the player (media.py) shows 3D videos instead.
"""
