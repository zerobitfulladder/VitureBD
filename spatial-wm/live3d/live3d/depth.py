"""depth.py — pluggable monocular-depth backends.

Each backend takes an RGB uint8 frame (HxWx3 numpy) and returns a normalized
depth map as float32 HxW in [0,1], larger = nearer (ready to upload as a GL
texture). DA2Depth (Depth Anything V2, torch fp16) is the default; a TensorRT
backend can slot in later behind the same interface.
"""
import os
import numpy as np
import torch
import torch.nn.functional as F

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# da2s.onnx / da2s_fp16.engine live in the nunif dir (sibling repo, built there)
_NUNIF = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "nunif"))


def _normalize(d, lo_hi=(2.0, 98.0)):
    lo = torch.quantile(d, lo_hi[0] / 100.0)
    hi = torch.quantile(d, lo_hi[1] / 100.0)
    return ((d - lo) / (hi - lo + 1e-6)).clamp(0.0, 1.0)


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
        depth = _normalize(depth)
        return depth.cpu().numpy().astype(np.float32)


def _export_da2_onnx(path, h, w):
    from transformers import AutoModelForDepthEstimation

    class _W(torch.nn.Module):
        def __init__(s):
            super().__init__()
            s.m = AutoModelForDepthEstimation.from_pretrained(
                "depth-anything/Depth-Anything-V2-Small-hf").eval()

        def forward(s, x):
            return s.m(pixel_values=x).predicted_depth

    print("[live3d] exporting DA2 -> ONNX (one-time)...", flush=True)
    torch.onnx.export(_W().eval(), (torch.randn(1, 3, h, w),), path,
                      input_names=["x"], output_names=["depth"], opset_version=17, dynamo=False)


def _build_engine(onnx_path, engine_path, logger):
    import tensorrt as trt
    print("[live3d] building TensorRT engine (one-time, ~5 min)...", flush=True)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            raise RuntimeError("TensorRT failed to parse " + onnx_path)
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    ser = builder.build_serialized_network(network, config)
    with open(engine_path, "wb") as f:
        f.write(ser)


class TRTDepth:
    """Depth Anything V2 Small via TensorRT FP16 (fixed 392x686). ~2.3x faster than
    torch-fp16. Loads a cached engine; builds/exports it one-time if missing."""
    name = "DA2-Small-TRT"
    H, W = 392, 686

    def __init__(self, device="cuda"):
        import tensorrt as trt
        self.dev = device
        onnx = os.path.join(_NUNIF, "da2s.onnx")
        engine = os.path.join(_NUNIF, "da2s_fp16.engine")
        logger = trt.Logger(trt.Logger.ERROR)
        if not os.path.exists(engine):
            if not os.path.exists(onnx):
                _export_da2_onnx(onnx, self.H, self.W)
            _build_engine(onnx, engine, logger)
        runtime = trt.Runtime(logger)
        with open(engine, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.in_name = self.engine.get_tensor_name(0)
        self.out_name = self.engine.get_tensor_name(self.engine.num_io_tensors - 1)
        self.out = torch.empty(tuple(self.engine.get_tensor_shape(self.out_name)),
                               device=device, dtype=torch.float32)
        self.mean = _MEAN.to(device)
        self.std = _STD.to(device)

    def reset(self):
        pass

    @torch.no_grad()
    def __call__(self, frame_rgb_u8):
        t = torch.from_numpy(frame_rgb_u8).to(self.dev).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        t = F.interpolate(t, size=(self.H, self.W), mode="bilinear", align_corners=False)
        t = ((t - self.mean) / self.std).contiguous()
        # keep preprocessing + TRT on the current stream (correct ordering)
        stream = torch.cuda.current_stream()
        self.ctx.set_tensor_address(self.in_name, t.data_ptr())
        self.ctx.set_tensor_address(self.out_name, self.out.data_ptr())
        self.ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        return _normalize(self.out.squeeze(0)).cpu().numpy().astype(np.float32)


def make_depth(args):
    """Factory — extend here for further backends."""
    if getattr(args, "trt", False):
        return TRTDepth()
    return DA2Depth(model_id=args.model, in_h=args.in_h)
