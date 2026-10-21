"""pose.py — the head pose, from the glasses' helper process to the session.

The helper polls VITURE's SDK about 1000 times a second and writes the latest
sample here; the session reads it once a frame, just before it places the
screen. It is one small block in /dev/shm guarded by a sequence count (a
seqlock): the writer makes the count odd while it writes and even after, and
a reader that saw it odd, or changed, reads again.

  seq u64 | t f64 | px py pz f64 | qw qx qy qz f64 | status i64

`t` is CLOCK_MONOTONIC (time.monotonic) when the sample was taken, the same
clock in both processes. The pose is the SDK's, predicted ahead by the lead
the session asked for: OpenGL axes (x right, y up, z backward), metres, and a
quaternion that turns head axes into world axes. `status` is 0 when tracking
is steady.
"""
import mmap
import os
import struct
from typing import NamedTuple

_SEQ = struct.Struct("<Q")
_DATA = struct.Struct("<d3d4dq")
SIZE = _SEQ.size + _DATA.size


class Sample(NamedTuple):
    t: float
    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float]     # w, x, y, z
    status: int


class PoseWriter:
    def __init__(self, path: str):
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.ftruncate(fd, SIZE)
            self._m = mmap.mmap(fd, SIZE)
        finally:
            os.close(fd)
        self.path = path
        self._seq = 0

    def write(self, t: float, pose, status: int) -> None:
        """`pose` is the SDK's 7 floats: px, py, pz, qw, qx, qy, qz."""
        self._seq += 1
        _SEQ.pack_into(self._m, 0, self._seq)
        _DATA.pack_into(self._m, _SEQ.size, t, *pose, status)
        self._seq += 1
        _SEQ.pack_into(self._m, 0, self._seq)

    def close(self) -> None:
        self._m.close()
        os.unlink(self.path)


class PoseReader:
    def __init__(self, path: str):
        fd = os.open(path, os.O_RDONLY)
        try:
            self._m = mmap.mmap(fd, SIZE, prot=mmap.PROT_READ)
        finally:
            os.close(fd)

    def read(self) -> Sample | None:
        """The latest sample, or None before the first."""
        for _ in range(100):
            seq = _SEQ.unpack_from(self._m, 0)[0]
            if seq & 1:
                continue
            v = _DATA.unpack_from(self._m, _SEQ.size)
            if _SEQ.unpack_from(self._m, 0)[0] != seq:
                continue
            if seq == 0:
                return None
            return Sample(v[0], v[1:4], v[4:8], v[8])
        return None

    def close(self) -> None:
        self._m.close()
