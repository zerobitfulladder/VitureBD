"""depth.py — from a desktop frame to depth and stereo offsets, on the GPU.

  frame -> depth model -> Stabilizer -> edge dilation -> mlbw (stereo.py)

Backends run a depth model on an RGB image at the model's fixed input size and
return its raw output (relative inverse depth: larger = nearer). DA2Depth runs
Depth Anything V2 in torch fp16; TRTDepth runs a prebuilt TensorRT engine of
the same model, about 2x faster.

The Stabilizer turns the raw maps into depth in [0, 1] that holds still:
  - the depth range (2nd-98th percentile) is smoothed over frames instead of
    taken fresh each frame, so the whole scene stops "breathing" when something
    near or bright comes into view; a cut resets it;
  - each pixel's depth is smoothed over frames in proportion to how still its
    colour is: unchanged pixels hold steady, pixels whose colour changed take
    the new depth at once, so moving edges don't trail.
Edge dilation (iw3's) then pulls the edges of near objects in by a few pixels,
so the background beside them isn't dragged along by the warp.

Results stay on the GPU: the renderer copies them into its textures through
the GL<->CUDA interop, or reads them back when that is unavailable. Once the
picture has held still long enough for the depth to settle, the pipeline
returns nothing until it changes: a still desktop costs no inference.
"""
import math
import threading
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from . import registry

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _trt_to_torch(trt_dtype):
    import tensorrt as trt
    return {trt.float32: torch.float32, trt.float16: torch.float16,
            trt.int32: torch.int32, trt.int8: torch.int8}[trt_dtype]


# --- backends ----------------------------------------------------------------

class DA2Depth:
    """Depth Anything V2 (HF), torch fp16."""

    def __init__(self, model="da2-small", device="cuda"):
        from transformers import AutoModelForDepthEstimation
        c = registry.cfg(model)
        self.in_size = (c["in_h"], c["in_w"])
        self.name = model
        self.model = (AutoModelForDepthEstimation.from_pretrained(c["hf_id"])
                      .to(device).half().eval())
        self.mean = _MEAN.to(device, torch.float16)
        self.std = _STD.to(device, torch.float16)

    @torch.no_grad()
    def raw(self, rgb):
        """rgb: (1, 3, H, W) float in [0, 1] at in_size. Returns (H, W) float."""
        t = (rgb.half() - self.mean) / self.std
        return self.model(pixel_values=t).predicted_depth.float().view(*self.in_size)


class TRTDepth:
    """Depth Anything V2 as a prebuilt TensorRT FP16 engine (engines.py)."""

    def __init__(self, model="da2-small", device="cuda"):
        import os
        import tensorrt as trt
        c = registry.cfg(model)
        self.in_size = (c["in_h"], c["in_w"])
        self.name = model + "-TRT"
        engine = registry.engine_path(model)
        if not os.path.exists(engine):
            raise FileNotFoundError(
                f"no engine at {engine}\n  build it first:  braindance-build-engine {model}")
        runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        with open(engine, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.in_name = self.engine.get_tensor_name(0)
        self.out_name = self.engine.get_tensor_name(self.engine.num_io_tensors - 1)
        # follow the engine's own I/O dtypes (fp16 engine -> fp16 buffers)
        self.in_dtype = _trt_to_torch(self.engine.get_tensor_dtype(self.in_name))
        self.out = torch.empty(tuple(self.engine.get_tensor_shape(self.out_name)),
                               device=device,
                               dtype=_trt_to_torch(self.engine.get_tensor_dtype(self.out_name)))
        self.mean = _MEAN.to(device)
        self.std = _STD.to(device)

    @torch.no_grad()
    def raw(self, rgb):
        t = ((rgb - self.mean) / self.std).to(self.in_dtype).contiguous()
        stream = torch.cuda.current_stream()
        self.ctx.set_tensor_address(self.in_name, t.data_ptr())
        self.ctx.set_tensor_address(self.out_name, self.out.data_ptr())
        self.ctx.execute_async_v3(stream.cuda_stream)
        return self.out.float().view(*self.in_size)


def build_depth(model_key, trt):
    """A depth backend by registry model name: the TensorRT engine (which must be
    built), or torch fp16 from the model's HF id."""
    return TRTDepth(model_key) if trt else DA2Depth(model_key)


# --- temporal stability (iw3's range EMA, and colour-gated per-pixel EMA) ------

class Stabilizer:
    QUANTILES = (0.02, 0.98)
    # Mean colour change of a 16x9 thumbnail above which a frame counts as a cut
    # (a workspace or window switch, a scene change). Scrolling text or a moving
    # window averages out far below it.
    CUT = 0.10
    # Per-pixel colour change (max over RGB, 0..1) at and above which a pixel
    # takes the new depth at once.
    MOTION = 0.06

    def __init__(self):
        self.range = self.depth = self.prev = self.thumb = None

    @torch.no_grad()
    def __call__(self, raw, rgb, range_ema, stabilize):
        """raw (H, W) model output; rgb (1, 3, H, W) in [0, 1]. range_ema: how much
        of the previous depth range carries over each frame (0 = per frame, iw3
        uses 0.75). stabilize: how much of a still pixel's previous depth carries
        over (0 = off). Returns a fresh (H, W) depth in [0, 1], larger = nearer."""
        thumb = F.adaptive_avg_pool2d(rgb, (9, 16))
        cut = (self.depth is None or self.depth.shape != raw.shape
               or bool((thumb - self.thumb).abs().mean() > self.CUT))
        self.thumb = thumb

        q = torch.quantile(raw[::4, ::4].flatten(),
                           torch.tensor(self.QUANTILES, device=raw.device))
        self.range = q if cut else torch.lerp(q, self.range, range_ema)
        lo, hi = self.range[0], self.range[1]
        d = ((raw - lo) / (hi - lo + 1e-6)).clamp_(0.0, 1.0)

        if cut:
            self.depth = d
        else:
            change = (rgb - self.prev).abs().amax(dim=1, keepdim=True)
            change = F.max_pool2d(change, 5, stride=1, padding=2)    # reach past moving edges
            floor = 1.0 - stabilize           # still pixels keep some of the new depth,
            a = floor + (1.0 - floor) * (change / self.MOTION).clamp_(0.0, 1.0)
            self.depth = torch.lerp(self.depth, d, a.view_as(d))   # so they still converge
        self.prev = rgb
        return self.depth.clone()


# --- edge dilation (iw3/dilation.py) -----------------------------------------

_BLUR = torch.tensor([[21, 31, 21], [31, 48, 31], [21, 31, 21]], dtype=torch.float32) / 256.0


def _edge_weight(x):
    """0..1 per pixel: how much the depth varies around it, normalised over the
    frame, so only edges are touched."""
    r = F.max_pool2d(x, 3, 1, 1) + F.max_pool2d(-x, 3, 1, 1)
    rc = r - r.mean()
    w = (rc / (rc.pow(2).mean().sqrt() + 1e-6)).clamp(-3, 3)
    return (w - w.amin()) / (w.amax() - w.amin() + 1e-6)


def dilate_edges(depth, n):
    """Grow the far side of every depth edge into the near side by about a pixel
    per pass (iw3 runs dilate_edge on the negated depth): `n` passes across and
    ceil(n/2) down, n = 2 being iw3's default. depth (H, W) -> new (H, W)."""
    x = -depth.view(1, 1, *depth.shape)
    blur = _BLUR.to(x.device).view(1, 1, 3, 3)
    nx, ny = n, math.ceil(n / 2)
    for kernel in [(3, 3)] * min(nx, ny) + [(1, 3)] * (nx - min(nx, ny)):
        w = _edge_weight(x)
        x2 = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), blur)
        x2 = F.max_pool2d(x2, kernel, 1, (kernel[0] // 2, kernel[1] // 2))
        x = x * (1 - w) + x2 * w
    return (-x).view_as(depth)


# --- the pipeline --------------------------------------------------------------

class Pipeline:
    """Desktop frames in, depth and stereo offsets out. `params` is read on each
    call (it may run on the worker thread): range_ema, stabilize, dilation,
    stereo (bool), disp (px), eye_w."""

    # A frame differs from the last when any 4x4 block's colour moved more than
    # this (so even a typed character counts). After SETTLED unchanged frames
    # with unchanged params the stabilized depth has converged: skip the models.
    STILL = 0.02
    SETTLED = 12

    def __init__(self, backend, stereo=None):
        self.backend = backend
        self.stereo = stereo                  # stereo.Stereo for backend.in_size, or None
        self.stabilizer = Stabilizer()
        self.params = SimpleNamespace(range_ema=0.75, stabilize=0.5, dilation=0,
                                      stereo=False, disp=18.0, eye_w=1920)
        self._lock = threading.Lock()         # the worker and the main thread may overlap
        self._thumb = self._key = None
        self._still = 0

    @property
    def size(self):
        return self.backend.in_size

    @property
    def settled(self):
        """The picture has held still long enough that the models stopped."""
        return self._still > self.SETTLED

    @torch.no_grad()
    def from_gl(self, rgba_u8):
        """`rgba_u8`: a CUDA (H, W, 4) uint8 frame read straight from a GL
        framebuffer (bottom-up)."""
        t = rgba_u8[:, :, :3].flip(0).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
        return self._run(t)

    @torch.no_grad()
    def from_frame(self, frame_u8):
        """`frame_u8`: an (H, W, 3) uint8 numpy frame, top row first."""
        t = torch.from_numpy(frame_u8).cuda().permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
        return self._run(t)

    def _run(self, rgb):
        """Returns depth (H, W) and, when params.stereo, flow (2, 4, H, W): both
        top row first, on the GPU, ready once this returns. None when nothing
        changed since the depth settled."""
        p = self.params
        rgb = F.interpolate(rgb, size=self.size, mode="bilinear", align_corners=False)
        with self._lock:
            thumb = F.avg_pool2d(rgb, 4)
            key = (p.range_ema, p.stabilize, p.dilation, p.stereo, p.disp, p.eye_w)
            still = (key == self._key and self._thumb is not None
                     and not bool((thumb - self._thumb).abs().amax() > self.STILL))
            self._thumb, self._key = thumb, key
            self._still = self._still + 1 if still else 0
            if self._still > self.SETTLED:
                return None
            raw = self.backend.raw(rgb)
            depth = self.stabilizer(raw, rgb, p.range_ema, p.stabilize)
            if p.dilation:
                depth = dilate_edges(depth, int(p.dilation))
            flow = (self.stereo(depth, p.disp, p.eye_w)
                    if p.stereo and self.stereo is not None else None)
            torch.cuda.current_stream().synchronize()
        return SimpleNamespace(depth=depth, flow=flow)


def to_numpy(t):
    return np.ascontiguousarray(t.cpu().numpy().astype(np.float32))
