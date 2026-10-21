"""glcuda.py — zero-copy GL<->CUDA interop, for the depth path and the
Gaussian splats' depth order.

Without it the depth path round-trips GPU->CPU->GPU twice: the composited frame
is read back to a numpy array and re-uploaded into torch, and the depth and the
stereo offsets are `.cpu()`d and re-uploaded into GL textures. That readback
also *stalls* the pipeline. This module registers the GL textures with CUDA so
the frame is read straight into a torch tensor and the results written straight
back to GL textures — no CPU, no stall.

Requires the GL context current on the calling thread (the main thread). The
tensors written may come from another thread, once that thread has synchronized
its stream. cuda-python (`cuda.bindings`) does the registration/copy; torch owns
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
    """Registers a float32 GL texture of `channels` channels (R32F or RGBA32F);
    copies a GPU (H, W) or (H, W, channels) float tensor straight into it — no
    host round-trip. Row 0 of the tensor is the texture's bottom row."""
    def __init__(self, glo, w, h, channels=1):
        self.w, self.h = w, h
        self.res = _ck(cudart.cudaGraphicsGLRegisterImage(
            glo, GL_TEXTURE_2D,
            cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard))
        self._row = w * 4 * channels   # float32

    def write(self, data):
        d = data.contiguous()
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


class GLBufferWriter:
    """Registers a GL buffer write-only; copies a GPU tensor's bytes into its
    start, no host round-trip."""
    def __init__(self, glo):
        self.res = _ck(cudart.cudaGraphicsGLRegisterBuffer(
            glo, cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard))

    def write(self, data):
        d = data.contiguous()
        _ck(cudart.cudaGraphicsMapResources(1, self.res, 0))
        ptr, size = _ck(cudart.cudaGraphicsResourceGetMappedPointer(self.res))
        _ck(cudart.cudaMemcpy(ptr, d.data_ptr(), min(size, d.numel() * d.element_size()),
                              cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice))
        _ck(cudart.cudaGraphicsUnmapResources(1, self.res, 0))

    def close(self):
        if self.res is not None:
            cudart.cudaGraphicsUnregisterResource(self.res)
            self.res = None
