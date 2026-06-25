"""build.py — turn a registered depth model into a runnable TensorRT engine.

Pipeline:  HuggingFace checkpoint --export--> ONNX --compile--> .engine (fp16)

Run once per machine. Engines are GPU- and TensorRT-version-specific, so they are
NOT committed (gitignored); rebuild after changing GPU or the tensorrt package.

  python models/build.py da2-small      # build one
  python models/build.py --all          # build everything in the registry
  python models/build.py da2-small --force   # re-export ONNX + rebuild engine

Requires this dir's venv (torch + transformers + tensorrt). See requirements.txt.
"""
import argparse
import os

import torch

import registry


def export_onnx(hf_id, path, h, w):
    # Export in FP16 on CUDA: TensorRT 11 is strongly-typed (no FP16 builder flag),
    # so the engine's precision is whatever the ONNX carries. fp16 ONNX -> fp16
    # engine. (fp16 CPU ops are largely unimplemented, hence .cuda().)
    from transformers import AutoModelForDepthEstimation

    class _W(torch.nn.Module):
        def __init__(s):
            super().__init__()
            s.m = AutoModelForDepthEstimation.from_pretrained(hf_id).eval()

        def forward(s, x):
            return s.m(pixel_values=x).predicted_depth

    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"[build] export {hf_id} -> {os.path.relpath(path)}  ({h}x{w}, fp16)", flush=True)
    model = _W().eval().half().cuda()
    dummy = torch.randn(1, 3, h, w, dtype=torch.float16, device="cuda")
    torch.onnx.export(model, (dummy,), path,
                      input_names=["x"], output_names=["depth"],
                      opset_version=17, dynamo=False)


def build_engine(onnx_path, engine_path):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    # TRT 11: strongly-typed network — precision comes from the (fp16) ONNX, not a
    # builder flag. Explicit batch is implied.
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("  ", parser.get_error(i))
            raise RuntimeError("ONNX parse failed for " + onnx_path)
    config = builder.create_builder_config()
    print(f"[build] compiling engine -> {os.path.relpath(engine_path)} (~minutes)", flush=True)
    ser = builder.build_serialized_network(network, config)
    if ser is None:
        raise RuntimeError("TensorRT engine build returned None")
    with open(engine_path, "wb") as f:
        f.write(ser)
    print(f"[build] done: {os.path.getsize(engine_path) // (1024 * 1024)} MB", flush=True)


def build(name, force=False):
    c = registry.cfg(name)
    onnx = registry.onnx_path(name)
    engine = registry.engine_path(name)
    if force or not os.path.exists(onnx):
        export_onnx(c["hf_id"], onnx, c["in_h"], c["in_w"])
    else:
        print(f"[build] reuse existing ONNX {os.path.relpath(onnx)}", flush=True)
    if force or not os.path.exists(engine):
        build_engine(onnx, engine)
    else:
        print(f"[build] engine already present {os.path.relpath(engine)} (use --force)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", nargs="?", help="model name from the registry")
    ap.add_argument("--all", action="store_true", help="build every registered model")
    ap.add_argument("--force", action="store_true", help="re-export ONNX and rebuild engine")
    a = ap.parse_args()
    names = list(registry.MODELS) if a.all else [a.model]
    if not names or names == [None]:
        ap.error("give a model name or --all. known: " + ", ".join(registry.MODELS))
    for n in names:
        build(n, force=a.force)


if __name__ == "__main__":
    main()
