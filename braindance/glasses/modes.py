"""modes.py — the display modes braindance drives the glasses in.

A mode is what the glasses scan out: one image for both eyes (2D), or one image
per eye side by side (3D, 3840 wide). Each carries the VITURE SDK code that
selects it. 1920x1200@120 is left out: the Luma Ultra (firmware 12.0.02.002)
accepts it but never comes up in it.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Mode:
    code: int          # VITURE_DISPLAY_MODE_*
    width: int
    height: int
    hz: int

    @property
    def sbs(self) -> bool:
        return self.width == 3840

    @property
    def desktop(self) -> tuple[int, int]:
        """The desktop's size: one eye in 3D, the whole panel in 2D."""
        return (self.width // 2 if self.sbs else self.width, self.height)

    @property
    def name(self) -> str:
        return f"{'3d' if self.sbs else '2d'}-{self.height}p{self.hz}"


MODES = (
    Mode(0x31, 1920, 1080, 60),
    Mode(0x33, 1920, 1080, 90),
    Mode(0x34, 1920, 1080, 120),
    Mode(0x41, 1920, 1200, 60),
    Mode(0x43, 1920, 1200, 90),
    Mode(0x32, 3840, 1080, 60),
    Mode(0x42, 3840, 1200, 60),
    Mode(0x45, 3840, 1200, 90),
)


def by_name(name: str) -> Mode:
    for m in MODES:
        if m.name == name:
            return m
    raise KeyError(f"unknown mode '{name}'. modes: {', '.join(m.name for m in MODES)}")


def by_code(code: int) -> Mode | None:
    return next((m for m in MODES if m.code == code), None)


def by_timing(width: int, height: int, hz: float) -> Mode | None:
    return next((m for m in MODES
                 if (m.width, m.height) == (width, height) and abs(m.hz - hz) < 1.5), None)
