"""inpaint.py — light_inpaint_v1: a network that paints the background a near
object uncovered, for the mesh warp's EDGE FILL.

The renderer finds the strips the mesh stretched across depth edges (render.
EDGE_FILL_FS). It packs both eyes, shrunk, with those strips as a mask, into
one texture; this network paints the masked pixels from what surrounds them,
and the renderer blends its picture back into each eye at full resolution,
only inside the strips (render.EDGE_MIX_FS).

The network is iw3's image inpainting model. It was trained with the holes on
the right of near objects, as the right eye has them; the left eye is packed
mirrored, as iw3 does. The architecture below is the inference path of nunif's
iw3/models/light_inpaint_v1.py and the nunif modules it uses (attention.py,
permute.py, norm.py, gaussian_filter.py), reduced to what the published
weights need; it loads nunif's checkpoint unchanged. One change: nunif pads
the input to the next multiple of 64 even when it is one already; this pads
only when needed, and the engine's size is a multiple of 64. That code is:

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


# --- the network (nunif iw3/models/light_inpaint_v1.py, inference only) ------

def _pixel_unshuffle(x, s):
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // s, s, W // s, s).permute(0, 1, 3, 5, 2, 4)
    return x.reshape(B, C * s * s, H // s, W // s)


def _pixel_shuffle(x, s):
    B, C, H, W = x.shape
    x = x.reshape(B, C // (s * s), s, s, H, W).permute(0, 1, 4, 2, 5, 3)
    return x.reshape(B, C // (s * s), H * s, W * s)


class _ReplicatePad(nn.Module):
    def __init__(self, padding):
        super().__init__()
        self.padding = padding

    def forward(self, x):
        return F.pad(x, self.padding, mode="replicate")


class _GMLP(nn.Module):
    """gMLP over the tokens of one window: a channel MLP gated by a learned
    mix across the window's positions. Returns its input added (the block that
    calls it adds the input again: that is how the weights were trained)."""
    def __init__(self, dim, seq_len, mlp_ratio):
        super().__init__()
        self.proj_in = nn.Linear(dim, dim * mlp_ratio * 2)
        self.proj_spatial = nn.Conv1d(seq_len, seq_len, kernel_size=1)
        self.proj_out = nn.Linear(dim * mlp_ratio, dim)

    def forward(self, x, norm1, norm2):
        y = F.gelu(self.proj_in(norm1(x)))
        u, v = y.chunk(2, dim=-1)
        return self.proj_out(u * self.proj_spatial(norm2(v))) + x


class _WindowGMLP2d(nn.Module):
    """_GMLP within square windows. A shifted block pads by half a window with
    zeros first, so its windows straddle the unshifted block's."""
    def __init__(self, dim, window, mlp_ratio, shift):
        super().__init__()
        self.window = window
        self.pad = window // 2 if shift else 0
        self.gmlp = _GMLP(dim, window * window, mlp_ratio)

    def forward(self, x, norm1, norm2):
        p = self.pad
        if p:
            x = F.pad(x, (p, p, p, p))
        B, C, H, W = x.shape
        s = self.window
        oh, ow = H // s, W // s
        x = x.reshape(B, C, oh, s, ow, s).permute(0, 2, 4, 3, 5, 1).reshape(B * oh * ow, s * s, C)
        x = self.gmlp(x, norm1, norm2)
        x = x.reshape(B, oh, ow, s, s, C).permute(0, 5, 1, 3, 2, 4).reshape(B, C, H, W)
        if p:
            x = x[:, :, p:H - p, p:W - p]
        return x


class _GLUConvMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pad = _ReplicatePad((1, 1, 1, 1))
        self.w1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.w2 = nn.Conv2d(dim // 2, dim, kernel_size=3)

    def forward(self, x):
        return self.w2(self.pad(F.glu(self.w1(x), dim=1)))


class _GMLPBlock(nn.Module):
    def __init__(self, dim, window, shift):
        super().__init__()
        self.gmlp = _WindowGMLP2d(dim, window, 2, shift)
        self.norm1 = nn.LayerNorm(dim, bias=False)
        self.norm2 = nn.LayerNorm(dim * 2, bias=False)
        self.glu_conv = _GLUConvMLP(dim)

    def forward(self, x):
        x = x + self.gmlp(x, self.norm1, self.norm2)
        return x + self.glu_conv(x)


def _gaussian_kernel(size):
    sigma = size * 0.15 + 0.35
    x = torch.linspace(-(size - 1) / 2, (size - 1) / 2, size)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


class LightInpaint(nn.Module):
    """inpaint.light_inpaint_v1. In: (B, 3, H, W) RGB in [0, 1] and (B, 1, H, W)
    mask, 1 = paint. Out: (B, 3, H, W), the network's picture everywhere (the
    caller keeps its own pixels outside the mask)."""
    SCALE = 4                   # pixel (un)shuffle factor
    MOD = 64                    # SCALE x the deepest window

    def __init__(self):
        super().__init__()
        C, C2, pack = 96, 192, self.SCALE ** 2
        self.mask_bias = nn.Parameter(torch.zeros(1, C, 1, 1))
        self.patch = nn.Sequential(nn.Conv2d(3 * pack, C, kernel_size=1), nn.LeakyReLU(0.2))
        self.enc1 = _GMLPBlock(C, 16, True)
        self.down = nn.Conv2d(C, C2, kernel_size=2, stride=2)
        self.enc2 = nn.Sequential(_GMLPBlock(C2, 8, False), _GMLPBlock(C2, 8, True),
                                  _GMLPBlock(C2, 8, False), _GMLPBlock(C2, 8, True))
        self.up = nn.Conv2d(C2, C * 4, kernel_size=1)
        self.dec1 = _GMLPBlock(C, 16, False)
        self.to_image = nn.Sequential(_ReplicatePad((1, 1, 1, 1)),
                                      nn.Conv2d(C, 3 * pack, kernel_size=3))
        k = _gaussian_kernel(15)
        self.register_buffer("blur_h", k.view(1, 1, 1, 15), persistent=False)
        self.register_buffer("blur_v", k.view(1, 1, 15, 1), persistent=False)

    def forward(self, x, mask):
        # iw3's preprocess: blank the masked pixels; the mask grows by a blur
        x = x * (1 - mask)
        m = F.pad(mask, (7, 7, 7, 7), mode="replicate")
        mask = torch.clamp(F.conv2d(F.conv2d(m, self.blur_h), self.blur_v) + mask, 0, 1)

        H, W = x.shape[2:]
        ph, pw = -H % self.MOD, -W % self.MOD
        x = (x - 0.5) / 0.5
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")
            mask = F.pad(mask, (0, pw, 0, ph), mode="replicate")
        x = self.patch(_pixel_unshuffle(x, self.SCALE))
        hole = _pixel_unshuffle(mask, self.SCALE).amax(dim=1, keepdim=True) > 0.99
        x = torch.where(hole, self.mask_bias.to(x.dtype), x)
        x1 = self.enc1(x)
        x2 = self.up(self.enc2(self.down(x1)))
        x = self.dec1(x1 + F.pixel_shuffle(x2, 2))
        x = _pixel_shuffle(self.to_image(x), self.SCALE)
        return x[:, :, :H, :W]


class BothEyes(nn.Module):
    """The exported graph. In: (2, 4, H, W) in [0, 1], the left eye mirrored then
    the right eye, RGB plus the mask (1 = paint). Out: (2, 3, H, W) float32, the
    network's picture, clamped, in the same layout."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        rgb, mask = x[:, :3], x[:, 3:]
        return self.net(rgb, mask).clamp(0, 1).float()


# --- weights, export, engine ------------------------------------------------

def weights_path() -> str:
    c = registry.INPAINT
    return os.path.join(registry.MODELS_DIR, c["dir"], c["weights"])


def download_weights() -> str:
    import urllib.request
    path = weights_path()
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        url = registry.INPAINT["url"] + registry.INPAINT["weights"]
        print(f"[build] download {url}", flush=True)
        urllib.request.urlretrieve(url, path + ".part")
        os.replace(path + ".part", path)
    return path


def load_net() -> LightInpaint:
    ck = torch.load(download_weights(), map_location="cpu", weights_only=True)
    net = LightInpaint()
    net.load_state_dict(ck["state_dict"])
    return net.eval()


def export_onnx(size: tuple[int, int], path: str) -> None:
    """fp16 ONNX of BothEyes for a fixed per-eye size (h, w)."""
    h, w = size
    net = BothEyes(load_net()).half().cuda().eval()
    x = torch.rand(2, 4, h, w, dtype=torch.float16, device="cuda")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"[build] export light_inpaint_v1 -> {os.path.relpath(path)}  (2x{h}x{w}, fp16)", flush=True)
    with torch.no_grad():
        torch.onnx.export(net, (x,), path, input_names=["x"], output_names=["rgb"],
                          opset_version=17, dynamo=False)


# --- runtime -------------------------------------------------------------------

class Inpaint:
    """The TensorRT engine of BothEyes at registry.INPAINT["size"] per eye."""

    def __init__(self):
        self.size = registry.INPAINT["size"]         # (h, w) per eye
        path = registry.inpaint_engine_path(self.size)
        if not os.path.exists(path):
            raise FileNotFoundError(
                "no inpaint engine\n  build it first:  braindance-build-engine inpaint")
        import tensorrt as trt
        runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        with open(path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        h, w = self.size
        self.x = torch.empty(2, 4, h, w, dtype=torch.float16, device="cuda")
        self.rgb = torch.empty(2, 3, h, w, dtype=torch.float32, device="cuda")
        self.out = torch.ones(h, 2 * w, 4, dtype=torch.float32, device="cuda")
        self.ctx.set_tensor_address("x", self.x.data_ptr())
        self.ctx.set_tensor_address("rgb", self.rgb.data_ptr())

    @torch.no_grad()
    def __call__(self, packed: torch.Tensor) -> torch.Tensor:
        """Paint a packed (h, 2w, 4) uint8 frame (both eyes side by side, RGB +
        mask, as the renderer packs it). Returns (h, 2w, 4) float32 in the same
        layout, alpha 1; the tensor is reused by the next call."""
        h, w = self.size
        self.x.copy_(packed.view(h, 2, w, 4).permute(1, 3, 0, 2))
        self.x.mul_(1.0 / 255.0)
        stream = torch.cuda.current_stream()
        self.ctx.execute_async_v3(stream.cuda_stream)
        self.out[..., :3].copy_(self.rgb.permute(2, 0, 3, 1).reshape(h, 2 * w, 3))
        stream.synchronize()
        return self.out
