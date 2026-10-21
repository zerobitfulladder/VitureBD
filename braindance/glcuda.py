"""glcuda.py — zero-copy GL<->CUDA interop for the depth path.

The sync depth path otherwise round-trips GPU->CPU->GPU twice: the composited
depth-source is read back to a numpy frame and re-uploaded into torch, and the
depth result is `.cpu()`d and re-uploaded into a GL texture. That readback also
*stalls* the pipeline. This module registers the GL textures with CUDA so the
frame is read straight into a torch tensor and the depth written straight back to
a GL texture — no CPU, no stall.

Requires the GL context current on the calling thread (so: the sync/main-thread
depth path). cuda-python (`cuda.bindings`) does the registration/copy; torch owns
the CUDA context the copies land in.
"""
import torch
from cuda.bindings import runtime as cudart

GL_TEXTURE_2D = 0x0DE1


def _ck(ret):
    """cuda-python calls return (err, *results); raise on error, unwrap results."""
    err, *rest = ret if isinstance(ret, (tuple, list)) else (ret,)
    if err != cudart.cudaError_t.cudaSuccess:
        name = cudart.cudaGetErrorString(err)[1]
        raise RuntimeError(f"CUDA interop error: {name}")
    return rest[0] if len(rest) == 1 else tuple(rest)


class GLTexReader:
    """Registers an RGBA8 GL texture read-only; copies it into a reusable GPU
    uint8 (H, W, 4) tensor each frame — no host round-trip."""
    def __init__(self, glo, w, h):
        self.w, self.h = w, h
        self.res = _ck(cudart.cudaGraphicsGLRegisterImage(
            glo, GL_TEXTURE_2D,
            cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsReadOnly))
        self.buf = torch.empty((h, w, 4), dtype=torch.uint8, device="cuda")
        self._row = w * 4

    def read(self):
        _ck(cudart.cudaGraphicsMapResources(1, self.res, 0))
        arr = _ck(cudart.cudaGraphicsSubResourceGetMappedArray(self.res, 0, 0))
        _ck(cudart.cudaMemcpy2DFromArray(
            self.buf.data_ptr(), self._row, arr, 0, 0, self._row, self.h,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice))
        _ck(cudart.cudaGraphicsUnmapResources(1, self.res, 0))
        return self.buf

    def close(self):
        if self.res is not None:
            cudart.cudaGraphicsUnregisterResource(self.res)
            self.res = None


class GLTexWriter:
    """Registers a single-channel float32 (R32F) GL texture; copies a GPU (H, W)
    float tensor straight into it — no host round-trip."""
    def __init__(self, glo, w, h):
        self.w, self.h = w, h
        self.res = _ck(cudart.cudaGraphicsGLRegisterImage(
            glo, GL_TEXTURE_2D,
            cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard))
        self._row = w * 4   # float32

    def write(self, depth_hw):
        d = depth_hw.contiguous()
        _ck(cudart.cudaGraphicsMapResources(1, self.res, 0))
        arr = _ck(cudart.cudaGraphicsSubResourceGetMappedArray(self.res, 0, 0))
        _ck(cudart.cudaMemcpy2DToArray(
            arr, 0, 0, d.data_ptr(), self._row, self._row, self.h,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice))
        _ck(cudart.cudaGraphicsUnmapResources(1, self.res, 0))

    def close(self):
        if self.res is not None:
            cudart.cudaGraphicsUnregisterResource(self.res)
            self.res = None
