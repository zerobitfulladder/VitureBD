"""stereo.py — mlbw_l2: a learned backward warp that makes each eye's view.

The network (iw3's sbs.mlbw_l2) looks only at the depth map, plus two constant
planes for 3D strength and convergence. For every pixel it predicts two
horizontal sampling offsets and a blend weight for each. The renderer does the
rest at full resolution: an eye's pixel is the weighted blend of the desktop
sampled at those two offsets (render.MLBW_FS), so there are no holes to fill.

One pass makes both eyes: the right eye is the left-eye network run on the
mirrored depth, mirrored back. Offsets come out in network pixels; `UV` turns
them into texture coordinates.

The architecture below is the inference path of nunif's iw3/models/mlbw.py and
the nunif modules it uses (attention.py, permute.py), reduced to what the
published weights need. It loads nunif's checkpoints unchanged. That code is:

  Copyright (C) 2019-2023 nagadomi <https://github.com/nagadomi/>

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import registry


# --- the network (nunif iw3/models/mlbw.py, inference only) ------------------

class _ReplicatePad(nn.Module):
    def __init__(self, padding):
        super().__init__()
        self.padding = padding

    def forward(self, x):
        return F.pad(x, self.padding, mode="replicate")


def _pixel_unshuffle(x, sh, sw):
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // sh, sh, W // sw, sw).permute(0, 1, 3, 5, 2, 4)
    return x.reshape(B, C * sh * sw, H // sh, W // sw)


def _pixel_shuffle(x, sh, sw):
    B, C, H, W = x.shape
    x = x.reshape(B, C // (sh * sw), sh, sw, H, W).permute(0, 1, 4, 2, 5, 3)
    return x.reshape(B, C // (sh * sw), H * sh, W * sw)


class _MHA(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.head_proj = nn.Linear(dim, dim)

    def forward(self, x, attn_mask):
        B, N, C = x.shape
        h = self.num_heads
        q, k, v = self.qkv_proj(x).split(C, dim=-1)
        q, k, v = (t.view(B, N, h, C // h).permute(0, 2, 1, 3) for t in (q, k, v))
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.head_proj(x.permute(0, 2, 1, 3).reshape(B, N, C))


class _WindowMHA2d(nn.Module):
    """Attention within 4x4 windows. A shifted block pads by half a window with
    zeros first, so its windows straddle the unshifted block's."""
    def __init__(self, dim, num_heads, window, shift):
        super().__init__()
        self.window = window
        self.pad = (window[0] // 2 if shift[0] else 0, window[1] // 2 if shift[1] else 0)
        self.mha = _MHA(dim, num_heads)

    def forward(self, x, attn_mask):
        ph, pw = self.pad
        if ph or pw:
            x = F.pad(x, (pw, pw, ph, ph))
        B, C, H, W = x.shape
        sh, sw = self.window
        oh, ow = H // sh, W // sw
        x = x.reshape(B, C, oh, sh, ow, sw).permute(0, 2, 4, 3, 5, 1).reshape(B * oh * ow, sh * sw, C)
        x = self.mha(x, attn_mask)
        x = x.reshape(B, oh, ow, sh, sw, C).permute(0, 5, 1, 3, 2, 4).reshape(B, C, H, W)
        if ph or pw:
            x = x[:, :, ph:H - ph, pw:W - pw]
        return x


class _WindowScoreBias(nn.Module):
    """A learned attention-score bias from the offset between window positions."""
    def __init__(self, window):
        super().__init__()
        n = window[0] * window[1]
        pos = torch.stack(torch.meshgrid(torch.arange(window[0]), torch.arange(window[1]),
                                         indexing="ij"), dim=2).reshape(n, 2)
        delta = [tuple(p) for p in (pos[:, None, :] - pos[None, :, :]).reshape(n * n, 2).tolist()]
        unique = sorted(set(delta))
        self.n = n
        self.register_buffer("index", torch.tensor([unique.index(d) for d in delta]))
        u = torch.tensor(unique, dtype=torch.float32)
        self.register_buffer("delta", u / u.abs().max())
        hidden = int(n ** 0.5) * 2
        self.to_bias = nn.Sequential(nn.Linear(2, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self):
        return self.to_bias(self.delta)[self.index].reshape(self.n, self.n)


class _WABlock(nn.Module):
    def __init__(self, dim, window, shift, num_heads):
        super().__init__()
        self.mha = _WindowMHA2d(dim, num_heads, window, shift)
        self.conv_mlp = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            _ReplicatePad((1, 1, 1, 1)),
            nn.Conv2d(dim, dim, kernel_size=3),
        )
        self.bias = _WindowScoreBias(window)

    def forward(self, x):
        x = x + self.mha(x, self.bias().to(x.dtype))
        return x + self.conv_mlp(x)


class MLBW(nn.Module):
    """sbs.mlbw_l2. In: (B, 3, H, W) = depth, divergence and convergence planes.
    Out: offsets (B, 2, H, W) in network pixels, and softmax weights (B, 2, H, W)."""
    LAYERS = 2
    PACK = (1, 8)               # pixel (un)shuffle factor
    MOD = 4

    def __init__(self):
        super().__init__()
        C = 32 * self.LAYERS
        pack = self.PACK[0] * self.PACK[1]
        self.lv1_in = nn.Sequential(
            _ReplicatePad((4, 4, 0, 0)),
            nn.Conv2d(3, C // pack, kernel_size=(1, 9)),
            nn.LeakyReLU(0.2),
        )
        self.lv2 = nn.Sequential(
            _WABlock(C, (4, 4), (True, True), self.LAYERS),
            _WABlock(C, (4, 4), (False, False), self.LAYERS),
            _WABlock(C, (4, 4), (True, True), self.LAYERS),
            _WABlock(C, (4, 4), (False, False), self.LAYERS),
        )
        self.lv1_out = nn.Sequential(
            _ReplicatePad((4, 4, 0, 0)),
            nn.Conv2d(C // pack, self.LAYERS * 2, kernel_size=(1, 9)),
        )

    def forward(self, x):
        H, W = x.shape[2:]
        pw = self.MOD * self.PACK[1] - W % (self.MOD * self.PACK[1])
        ph = self.MOD * self.PACK[0] - H % (self.MOD * self.PACK[0])
        pw1, ph1 = pw // 2, ph // 2
        x = F.pad(x, (pw1, pw - pw1, ph1, ph - ph1), mode="replicate")
        x = x1 = self.lv1_in(x)
        x = _pixel_unshuffle(x, *self.PACK)
        x = self.lv2(x)
        x = _pixel_shuffle(x, *self.PACK)
        x = self.lv1_out(x + x1)
        x = x[:, :, ph1:ph1 + H, pw1:pw1 + W]
        delta, weight = x.chunk(2, dim=1)
        return delta.float(), F.softmax(weight.float(), dim=1)


class BothEyes(nn.Module):
    """The exported graph. In: depth (1, 1, H, W) in [0, 1], larger = nearer, and
    feat (1, 2, 1, 1) = divergence and convergence features. Out: (2, 4, H, W),
    left eye then right: offsets d0, d1 and weights w0, w1. The left eye samples
    at x + d, the right eye at x - d (its offsets are mirrored back)."""
    def __init__(self, mlbw):
        super().__init__()
        self.mlbw = mlbw

    def forward(self, depth, feat):
        x = torch.cat([depth, feat.expand(1, 2, depth.shape[2], depth.shape[3])], dim=1)
        x = torch.cat([x, x.flip(3)], dim=0)
        delta, weight = self.mlbw(x)
        out = torch.cat([delta, weight], dim=1)
        return torch.cat([out[:1], out[1:].flip(3)], dim=0)


def features(disp_px: float, eye_w: int, net_w: int) -> tuple[float, float]:
    """The network's two constant inputs for braindance's 3D strength `disp_px`
    (per-eye shift = disp_px * (depth - 0.5) at the eye's width). In iw3 terms
    that is convergence 0.5 and a divergence of disp_px eye pixels; the network
    takes it in its own pixels, over 32."""
    div = disp_px * net_w / eye_w
    return div / 32.0, -div * 0.5 / 32.0


def level(disp_px: float, eye_w: int) -> int:
    """Which trained weights to use: iw3 has one set per divergence range, in
    percent of the width (<= 4, <= 7, above)."""
    divergence = disp_px / (0.005 * eye_w)
    return 1 if divergence <= 4 else 2 if divergence <= 7 else 3


def uv_scale(net_w: int) -> float:
    """Network pixels -> texture coordinates (grid_sample, align_corners=True)."""
    return 1.0 / (2 * (net_w // 2 - 1))


# --- weights, export, engines ----------------------------------------------

def weights_path(lvl: int) -> str:
    c = registry.STEREO
    return os.path.join(registry.MODELS_DIR, c["dir"], c["weights"][lvl])


def download_weights(lvl: int) -> str:
    import urllib.request
    path = weights_path(lvl)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        url = registry.STEREO["url"] + registry.STEREO["weights"][lvl]
        print(f"[build] download {url}", flush=True)
        urllib.request.urlretrieve(url, path + ".part")
        os.replace(path + ".part", path)
    return path


def load_mlbw(lvl: int) -> MLBW:
    ck = torch.load(download_weights(lvl), map_location="cpu", weights_only=True)
    m = MLBW()
    m.load_state_dict(ck["state_dict"])
    return m.eval()


def export_onnx(lvl: int, size: tuple[int, int], path: str) -> None:
    """fp16 ONNX of BothEyes for a fixed depth size (h, w): TensorRT 11 is strongly
    typed, so the engine runs in whatever precision the ONNX carries."""
    h, w = size
    net = BothEyes(load_mlbw(lvl)).half().cuda().eval()
    depth = torch.rand(1, 1, h, w, dtype=torch.float16, device="cuda")
    feat = torch.tensor(features(18.0, 1920, w), dtype=torch.float16,
                        device="cuda").view(1, 2, 1, 1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"[build] export mlbw_l2 d{lvl} -> {os.path.relpath(path)}  ({h}x{w}, fp16)", flush=True)
    with torch.no_grad():
        torch.onnx.export(net, (depth, feat), path, input_names=["depth", "feat"],
                          output_names=["flow"], opset_version=17, dynamo=False)


# --- runtime -------------------------------------------------------------------

class Stereo:
    """TensorRT engines of BothEyes, one per weight level, loaded on first use.
    They run at registry.STEREO["size"]; other depth maps are resized to it."""

    def __init__(self):
        self.size = registry.STEREO["size"]          # (h, w) of the offsets it makes
        self.uv = uv_scale(self.size[1])
        self._engines = {}
        for lvl in registry.STEREO["weights"]:
            if not os.path.exists(registry.stereo_engine_path(lvl, self.size)):
                raise FileNotFoundError(
                    "no mlbw engines\n  build them first:  braindance-build-engine mlbw")

    def _engine(self, lvl):
        e = self._engines.get(lvl)
        if e is None:
            import tensorrt as trt
            runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
            with open(registry.stereo_engine_path(lvl, self.size), "rb") as f:
                engine = runtime.deserialize_cuda_engine(f.read())
            ctx = engine.create_execution_context()
            h, w = self.size
            e = self._engines[lvl] = (
                engine, ctx,
                torch.empty(1, 1, h, w, dtype=torch.float16, device="cuda"),
                torch.empty(1, 2, 1, 1, dtype=torch.float16, device="cuda"),
                torch.empty(2, 4, h, w, dtype=torch.float32, device="cuda"))
            ctx.set_tensor_address("depth", e[2].data_ptr())
            ctx.set_tensor_address("feat", e[3].data_ptr())
            ctx.set_tensor_address("flow", e[4].data_ptr())
        return e

    @torch.no_grad()
    def __call__(self, depth_hw: torch.Tensor, disp_px: float, eye_w: int) -> torch.Tensor:
        """Both eyes' offsets and weights (2, 4, h, w) at self.size, for a
        top-down depth map of any size. Returns a fresh tensor."""
        lvl = min(level(disp_px, eye_w), max(registry.STEREO["weights"]))
        _, ctx, din, fin, out = self._engine(lvl)
        d = depth_hw.view(1, 1, *depth_hw.shape)
        if tuple(depth_hw.shape) != self.size:
            d = F.interpolate(d, size=self.size, mode="bilinear", antialias=True,
                              align_corners=False)
        din.copy_(d)
        f0, f1 = features(disp_px, eye_w, self.size[1])
        fin.copy_(torch.tensor((f0, f1), dtype=torch.float16).view(1, 2, 1, 1))
        stream = torch.cuda.current_stream()
        ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        return out.clone()
