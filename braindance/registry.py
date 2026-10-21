"""registry.py — the catalog of models braindance knows how to build & run: the
depth models, the stereo warp network, and the inpainting network.

Both engines.py (build-time: weights -> ONNX -> .engine) and depth.py /
stereo.py / inpaint.py (run-time: load the .engine) read this. Paths are
relative to the project's models/ dir, which holds the built artifacts. Engine input is a FIXED size (multiple of 14 for the
DA2 ViT patch grid); frames are resized to it before inference. Tune in_h/in_w
per model here.
"""
import os

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

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


# The stereo warp network (stereo.py): iw3's mlbw_l2, one set of weights per
# divergence level. braindance's 3D strength tops out at 60 px, iw3 divergence
# 6.25 at 1920 wide, so levels 1 (<= 4) and 2 (<= 7) cover it. It runs at one
# size: about 9.5 ms for both eyes on an RTX 3060 Laptop at 392x686, against
# 35 ms at da2-small-hi's 700x1232, so larger depth maps are shrunk for it.
STEREO = {
    "dir": "mlbw-l2",
    "url": "https://github.com/nagadomi/nunif/releases/download/0.0.0/",
    "weights": {1: "iw3_mlbw_l2_d1_20250627.pth", 2: "iw3_mlbw_l2_d2_20250627.pth"},
    "size": (392, 686),
}


# The inpainting network (inpaint.py): iw3's light_inpaint_v1, for the mesh's
# EDGE FILL. It runs on both eyes at once, each shrunk to `size`, a multiple of
# 64 so it needs no padding. On an RTX 3060 Laptop both eyes take 15 ms at
# 512x896: a SYNC frame with da2-small that runs it takes 27 ms (p95 30), so it
# holds 30 fps. 576x1024 (17 ms) put the p50 frame at 31 ms, too near 33.3.
INPAINT = {
    "dir": "light-inpaint",
    "url": "https://github.com/nagadomi/nunif/releases/download/0.0.0/",
    "weights": "iw3_light_inpaint_v1_20250919.pth",
    "size": (512, 896),
}


def inpaint_onnx_path(size):
    return os.path.join(MODELS_DIR, INPAINT["dir"], f"{size[0]}x{size[1]}.onnx")


def inpaint_engine_path(size):
    return os.path.join(MODELS_DIR, INPAINT["dir"], f"{size[0]}x{size[1]}_fp16.engine")


def stereo_onnx_path(level, size):
    return os.path.join(MODELS_DIR, STEREO["dir"], f"d{level}_{size[0]}x{size[1]}.onnx")


def stereo_engine_path(level, size):
    return os.path.join(MODELS_DIR, STEREO["dir"], f"d{level}_{size[0]}x{size[1]}_fp16.engine")


def cfg(name):
    if name not in MODELS:
        raise KeyError(f"unknown model '{name}'. known: {', '.join(MODELS)}")
    return MODELS[name]


def onnx_path(name):
    return os.path.join(MODELS_DIR, cfg(name)["onnx"])


def engine_path(name):
    return os.path.join(MODELS_DIR, cfg(name)["engine"])
