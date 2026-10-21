"""depth.py — pluggable monocular-depth backends.

Each backend takes an RGB uint8 frame (HxWx3 numpy) and returns a normalized
depth map as float32 HxW in [0,1], larger = nearer (ready to upload as a GL
texture). DA2Depth (Depth Anything V2, torch fp16) is the default; a TensorRT
backend can slot in later behind the same interface.
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Models + their built engines live in braindance/models/ (self-contained, no nunif).
# registry.py there maps a model name -> hf_id, fixed input size, onnx/engine paths.
_MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, _MODELS_DIR)
import registry  # noqa: E402  (from models/)


def _trt_to_torch(trt_dtype):
    import tensorrt as trt
    return {trt.float32: torch.float32, trt.float16: torch.float16,
            trt.int32: torch.int32, trt.int8: torch.int8}[trt_dtype]


def _normalize(d, lo_hi=(2.0, 98.0)):
    lo = torch.quantile(d, lo_hi[0] / 100.0)
    hi = torch.quantile(d, lo_hi[1] / 100.0)
    return ((d - lo) / (hi - lo + 1e-6)).clamp(0.0, 1.0)


def _ema_apply(obj, d):
    """Temporal EMA on a torch depth tensor (in-place on the kept state) to damp
    frame-to-frame flicker/boiling. `obj` carries `.ema_alpha` (1.0 = off / fully
    responsive; ->0 = steadier but laggier) and `._ema`. Trades some ghosting on
    motion for stability on static content."""
    a = obj.ema_alpha
    if a >= 0.999:
        obj._ema = None
        return d
    if obj._ema is None or obj._ema.shape != d.shape:
        obj._ema = d.detach().clone()
    else:
        obj._ema.mul_(1.0 - a).add_(d, alpha=a)
    return obj._ema


class DA2Depth:
    """Depth Anything V2 (HF), fp16, manual (PIL-free) preprocessing."""
    name = "DA2-Small"

    def __init__(self, model_id="depth-anything/Depth-Anything-V2-Small-hf",
                 in_h=392, device="cuda", half=True):
        from transformers import AutoModelForDepthEstimation
        self.dev = device
        self.dtype = torch.float16 if half else torch.float32
        self.model = (AutoModelForDepthEstimation.from_pretrained(model_id)
                      .to(device).to(self.dtype).eval())
        self.in_h = in_h
        self.mean = _MEAN.to(device, self.dtype)
        self.std = _STD.to(device, self.dtype)
        self.name = model_id.split("/")[-1]
        self.ema_alpha = 1.0      # temporal EMA blend (1.0 = off); set live from app
        self._ema = None

    @torch.no_grad()
    def __call__(self, frame_rgb_u8):
        h, w, _ = frame_rgb_u8.shape
        in_w = max(14, int(round(self.in_h * w / h / 14)) * 14)
        t = torch.from_numpy(frame_rgb_u8).to(self.dev)
        t = t.permute(2, 0, 1).unsqueeze(0).to(self.dtype) / 255.0
        t = torch.nn.functional.interpolate(t, size=(self.in_h, in_w),
                                             mode="bilinear", align_corners=False)
        t = (t - self.mean) / self.std
        depth = self.model(pixel_values=t).predicted_depth.float().squeeze(0)  # HxW
        depth = _ema_apply(self, _normalize(depth))
        return depth.cpu().numpy().astype(np.float32)


class TRTDepth:
    """Depth Anything V2 via TensorRT FP16. ~2x faster than torch-fp16. Loads a
    prebuilt engine from models/ — run `python models/build.py <name>` first if
    it's missing (engines are GPU/TRT-version specific, so not committed)."""

    def __init__(self, model="da2-small", device="cuda"):
        import tensorrt as trt
        self.dev = device
        self.name = model + "-TRT"
        c = registry.cfg(model)
        self.H, self.W = c["in_h"], c["in_w"]
        engine = registry.engine_path(model)
        if not os.path.exists(engine):
            raise FileNotFoundError(
                f"no engine at {engine}\n  build it first:  python models/build.py {model}")
        logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(logger)
        with open(engine, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.in_name = self.engine.get_tensor_name(0)
        self.out_name = self.engine.get_tensor_name(self.engine.num_io_tensors - 1)
        # follow the engine's own I/O dtypes (fp16 engine -> fp16 buffers)
        self.in_dtype = _trt_to_torch(self.engine.get_tensor_dtype(self.in_name))
        out_dtype = _trt_to_torch(self.engine.get_tensor_dtype(self.out_name))
        self.out = torch.empty(tuple(self.engine.get_tensor_shape(self.out_name)),
                               device=device, dtype=out_dtype)
        self.mean = _MEAN.to(device)
        self.std = _STD.to(device)
        # (H_out, W_out) of the depth map — the GL interop depth texture is sized to it
        self.out_size = (int(self.out.shape[-2]), int(self.out.shape[-1]))
        self.ema_alpha = 1.0      # temporal EMA blend (1.0 = off); set live from app
        self._ema = None

    def reset(self):
        pass

    @torch.no_grad()
    def infer_gpu(self, rgba_u8):
        """Zero-copy path: `rgba_u8` is a CUDA (H, W, 4) uint8 tensor read straight
        from the GL framebuffer (bottom-up). Returns the (H_out, W_out) f32 depth in
        [0,1], bottom-up — ready to copy straight back into the GL depth texture. No
        host round-trip; the orientation flips that were numpy `[::-1]` run on GPU."""
        t = rgba_u8[:, :, :3].flip(0)                 # drop alpha; flip to upright
        t = t.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
        t = F.interpolate(t, size=(self.H, self.W), mode="bilinear", align_corners=False)
        t = ((t - self.mean) / self.std).to(self.in_dtype).contiguous()
        stream = torch.cuda.current_stream()
        self.ctx.set_tensor_address(self.in_name, t.data_ptr())
        self.ctx.set_tensor_address(self.out_name, self.out.data_ptr())
        self.ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        d = _ema_apply(self, _normalize(self.out.squeeze().float()))   # (H_out, W_out)
        return d.flip(0).contiguous()                 # back to bottom-up for the texture

    @torch.no_grad()
    def __call__(self, frame_rgb_u8):
        t = torch.from_numpy(frame_rgb_u8).to(self.dev).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        t = F.interpolate(t, size=(self.H, self.W), mode="bilinear", align_corners=False)
        t = ((t - self.mean) / self.std).to(self.in_dtype).contiguous()
        # keep preprocessing + TRT on the current stream (correct ordering)
        stream = torch.cuda.current_stream()
        self.ctx.set_tensor_address(self.in_name, t.data_ptr())
        self.ctx.set_tensor_address(self.out_name, self.out.data_ptr())
        self.ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        d = _ema_apply(self, _normalize(self.out.squeeze(0).float()))
        return d.cpu().numpy().astype(np.float32)


def build_depth(model_key, trt, in_h=392):
    """Build a depth backend by registry model name. trt=True -> TensorRT engine
    (must be built), else torch-fp16 from the model's HF id. Used both at startup
    and for LIVE backend swaps from the settings panel."""
    if trt:
        return TRTDepth(model=model_key)
    return DA2Depth(model_id=registry.cfg(model_key)["hf_id"], in_h=in_h)


def make_depth(args):
    """Startup factory from CLI args."""
    return build_depth(getattr(args, "trt_model", "da2-small"),
                       getattr(args, "trt", False), args.in_h)
