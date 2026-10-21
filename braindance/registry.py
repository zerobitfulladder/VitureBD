"""registry.py — the catalog of depth models live3d knows how to build & run.

Both models/build.py (build-time: HF -> ONNX -> .engine) and live3d/depth.py
(run-time: load the .engine) read this. Paths are relative to this models/ dir.
Engine input is a FIXED size (multiple of 14 for the DA2 ViT patch grid); frames
are resized to it before inference. Tune in_h/in_w per model here.
"""
import os

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS = {
    "da2-small": {
        "hf_id": "depth-anything/Depth-Anything-V2-Small-hf",
        "in_h": 392, "in_w": 686,
        "onnx": "da2-small/da2s.onnx",
        "engine": "da2-small/da2s_fp16.engine",
    },
    "da2-small-hi": {        # same model, ~1.8x resolution -> sharper edges, slower
        "hf_id": "depth-anything/Depth-Anything-V2-Small-hf",
        "in_h": 700, "in_w": 1232,
        "onnx": "da2-small-hi/da2s.onnx",
        "engine": "da2-small-hi/da2s_fp16.engine",
    },
    # add e.g. "da2-base", "video-depth-anything-small" here later
}


def cfg(name):
    if name not in MODELS:
        raise KeyError(f"unknown model '{name}'. known: {', '.join(MODELS)}")
    return MODELS[name]


def onnx_path(name):
    return os.path.join(MODELS_DIR, cfg(name)["onnx"])


def engine_path(name):
    return os.path.join(MODELS_DIR, cfg(name)["engine"])
