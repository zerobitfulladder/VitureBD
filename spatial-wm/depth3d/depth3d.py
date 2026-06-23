#!/usr/bin/env python3
"""depth3d — prototype: turn a 2D video into stereo 3D via monocular depth.

Pipeline per frame:
  1. decode a video frame (OpenCV)
  2. estimate depth with Depth Anything V2 (Small) on the GPU
  3. mesh-warp DIBR: render the frame as a grid whose vertices are shifted in X by
     disparity = f(depth), once per eye (the rasterizer stretches over occlusions
     so there are no holes)
  4. present side-by-side, fullscreen, to the glasses (3840x1080 in 3D SBS mode)

This validates the 3D quality / comfort / FPS before porting into the C++
stereo-mirror app. It reads a VIDEO FILE (not the screen) to avoid a capture
feedback loop and to test on controlled content.

Controls (focus the GL window):
  -, =      disparity  (3D strength) down / up
  [, ]      convergence (screen plane) back / forward
  ,, .      temporal smoothing  less / more
  s         swap left/right eye
  m         toggle mono preview (same frame both eyes, depth off)
  space     pause
  r         reset tuning
  q / Esc   quit
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
import glfw
import moderngl
from PIL import Image, ImageDraw, ImageFont
from Xlib import display as Xdisplay, X as Xlib_X, XK

# ---- depth model -----------------------------------------------------------

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

class Depth:
    """Depth Anything V2 (HF) — per-frame, fast, but flickers across frames."""
    def __init__(self, model_id, in_h, device="cuda", half=True):
        from transformers import AutoModelForDepthEstimation     # lazy (HF-only dep)
        self.dev = device
        self.dtype = torch.float16 if half else torch.float32
        self.model = (AutoModelForDepthEstimation.from_pretrained(model_id)
                      .to(device).to(self.dtype).eval())
        self.in_h = in_h
        self.mean = IMAGENET_MEAN.to(device, self.dtype)
        self.std  = IMAGENET_STD.to(device, self.dtype)

    def reset(self):
        pass   # stateless

    @torch.no_grad()
    def __call__(self, frame_rgb_u8):
        # frame_rgb_u8: HxWx3 uint8 (RGB) numpy
        h, w, _ = frame_rgb_u8.shape
        in_w = max(14, int(round(self.in_h * w / h / 14)) * 14)
        t = torch.from_numpy(frame_rgb_u8).to(self.dev)
        t = t.permute(2, 0, 1).unsqueeze(0).to(self.dtype) / 255.0          # 1x3xHxW
        t = torch.nn.functional.interpolate(t, size=(self.in_h, in_w),
                                            mode="bilinear", align_corners=False)
        t = (t - self.mean) / self.std
        depth = self.model(pixel_values=t).predicted_depth                 # 1xhxw (rel, larger=nearer)
        depth = depth.float().squeeze(0)                                   # hxw
        return depth                                                       # GPU tensor


def normalize_depth(d, lo_hi=(2.0, 98.0)):
    """Per-frame robust min-max to 0..1 (larger = nearer)."""
    lo = torch.quantile(d, lo_hi[0] / 100.0)
    hi = torch.quantile(d, lo_hi[1] / 100.0)
    return ((d - lo) / (hi - lo + 1e-6)).clamp(0.0, 1.0)


class IW3Depth:
    """Any iw3 depth model by name, loaded from the cloned nunif repo (no GUI deps):
      Any_V3_Mono              Depth Anything 3 (Large) — best quality, slow
      VDA_Stream_S/_B/_L       Video Depth Anything (streaming) — temporally stable
      ZoeD_* / DepthPro / ...  others iw3 supports
    reset_state() is a no-op for per-frame models and clears the window for VDA."""
    def __init__(self, model_name, in_h, device="cuda"):
        nunif_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "nunif"))
        if nunif_dir not in sys.path:
            sys.path.insert(0, nunif_dir)
        from iw3.depth_model_factory import create_depth_model
        self.dev = device
        self.m = create_depth_model(model_name)
        self.m.load(gpu=0, resolution=in_h)
        self.m.reset_state()

    def reset(self):
        self.m.reset_state()

    @torch.no_grad()
    def __call__(self, frame_rgb_u8):
        x = torch.from_numpy(frame_rgb_u8).permute(2, 0, 1).float().div(255).to(self.dev)
        d = self.m.infer(x)            # disparity (larger = nearer), some HxW
        while d.ndim > 2:
            d = d[0]
        return d


# ---- GL mesh-warp renderer -------------------------------------------------

VERT = """
#version 330
in vec2 in_uv;
uniform sampler2D depth_tex;
uniform float disp;       // max disparity, NDC x units
uniform float conv;       // convergence depth (0..1) -> zero disparity
uniform float eye_sign;   // +1 / -1
out vec2 v_uv;
void main() {
    float d = texture(depth_tex, in_uv).r;            // 0..1, larger = nearer
    float shift = eye_sign * disp * (d - conv);
    vec2 p = in_uv * 2.0 - 1.0;                        // uv -> NDC
    p.x += shift;
    gl_Position = vec4(p.x, -p.y, 0.0, 1.0);          // flip Y (texture is top-down)
    v_uv = in_uv;
}
"""

FRAG = """
#version 330
in vec2 v_uv;
uniform sampler2D color_tex;
out vec4 f;
void main() { f = texture(color_tex, v_uv); }
"""

HUD_VERT = """
#version 330
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() { gl_Position = vec4(in_pos, 0.0, 1.0); v_uv = in_uv; }
"""

HUD_FRAG = """
#version 330
in vec2 v_uv;
uniform sampler2D hud_tex;
out vec4 f;
void main() { f = texture(hud_tex, v_uv); }
"""

def make_grid(gx, gy):
    xs = np.linspace(0.0, 1.0, gx + 1, dtype="f4")
    ys = np.linspace(0.0, 1.0, gy + 1, dtype="f4")
    uv = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)          # (gx+1)*(gy+1) x 2
    idx = []
    stride = gx + 1
    for j in range(gy):
        for i in range(gx):
            a = j * stride + i; b = a + 1; c = a + stride; d = c + 1
            idx += [a, b, c, b, d, c]
    return uv.astype("f4"), np.array(idx, dtype="i4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="path to a 2D video file")
    ap.add_argument("--model", default="depth-anything/Depth-Anything-V2-Small-hf")
    ap.add_argument("--iw3", default=None,
                    help="use an iw3 model by name: Any_V3_Mono (DA3), VDA_Stream_S/_B/_L, ZoeD_*, DepthPro")
    ap.add_argument("--vda", action="store_true", help="shortcut for --iw3 VDA_Stream_S")
    ap.add_argument("--vda-model", default="VDA_Stream_S", help="VDA model when --vda is used")
    ap.add_argument("--in-h", type=int, default=392, help="depth model input height (mult of 14)")
    ap.add_argument("--disp", type=float, default=18.0, help="max disparity in px (per eye)")
    ap.add_argument("--conv", type=float, default=0.5, help="convergence depth 0..1")
    ap.add_argument("--smooth", type=float, default=0.6, help="temporal EMA (0=off..0.95)")
    ap.add_argument("--grid", type=int, default=256, help="warp grid columns")
    ap.add_argument("--max-w", type=int, default=1920, help="downscale frames wider than this")
    ap.add_argument("--panel", default="3840x1080")
    args = ap.parse_args()

    pw, ph = (int(x) for x in args.panel.split("x"))
    eye_w = pw // 2

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"cannot open {args.video}")

    print("loading depth model…", flush=True)
    iw3_name = args.iw3 or (args.vda_model if args.vda else None)
    if iw3_name:
        depth = IW3Depth(iw3_name, args.in_h)
        model_label = iw3_name
    else:
        depth = Depth(args.model, args.in_h)
        model_label = args.model.split("/")[-1]

    if not glfw.init():
        sys.exit("glfw init failed")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.DECORATED, glfw.FALSE)
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)       # create unmapped; map after override-redirect
    mon = glfw.get_primary_monitor()
    mode = glfw.get_video_mode(mon)
    pw, ph = mode.size.width, mode.size.height       # full panel (3840x1080 in 3D mode)
    eye_w = pw // 2

    # A normal window gets tiled/confined by qtile (fake_screens caps it at 1920).
    # Create it HIDDEN, set override-redirect while still unmapped (so the WM never
    # manages it) and size it to the whole panel, THEN map. Doing this pre-map avoids
    # the tile-then-fix race that made it flaky on restart.
    win = glfw.create_window(pw, ph, "depth3d", None, None)
    if not win:
        glfw.terminate(); sys.exit("window failed")
    glfw.make_context_current(win)

    xdpy = Xdisplay.Display()
    xwin = xdpy.create_resource_object("window", glfw.get_x11_window(win))
    xwin.change_attributes(override_redirect=True)   # set while unmapped
    xwin.configure(x=0, y=0, width=pw, height=ph)
    xdpy.sync()
    glfw.show_window(win)                            # maps as override-redirect
    xwin.configure(stack_mode=Xlib_X.Above)
    xdpy.sync()
    glfw.swap_interval(1)

    ctx = moderngl.create_context()
    print(f"[depth3d] panel {pw}x{ph}  framebuffer {glfw.get_framebuffer_size(win)}", flush=True)

    # Keyboard via XQueryKeymap (override-redirect windows don't reliably get GLFW
    # key events under qtile's follow-mouse focus).
    def key_down(keysym):
        kc = xdpy.keysym_to_keycode(keysym)
        if not kc:
            return False
        km = xdpy.query_keymap()
        return bool(km[kc >> 3] & (1 << (kc & 7)))

    prog = ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
    gy = max(1, args.grid * ph // pw)
    uv, idx = make_grid(args.grid, gy)
    vbo = ctx.buffer(uv.tobytes())
    ibo = ctx.buffer(idx.tobytes())
    vao = ctx.vertex_array(prog, [(vbo, "2f", "in_uv")], ibo)
    prog["color_tex"] = 0
    prog["depth_tex"] = 1

    # On-screen stats HUD (the terminal is hidden behind the fullscreen overlay).
    HUD_W, HUD_H = 640, 40
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    hud_prog = ctx.program(vertex_shader=HUD_VERT, fragment_shader=HUD_FRAG)
    hud_prog["hud_tex"] = 0
    hwn, hhn = (HUD_W / eye_w) * 2.0, (HUD_H / ph) * 2.0
    qx0, qy1 = -0.97, 0.97
    qx1, qy0 = qx0 + hwn, qy1 - hhn
    quad = np.array([qx0, qy1, 0, 0,  qx1, qy1, 1, 0,  qx0, qy0, 0, 1,
                     qx1, qy1, 1, 0,  qx1, qy0, 1, 1,  qx0, qy0, 0, 1], dtype="f4")
    hud_vao = ctx.vertex_array(hud_prog, [(ctx.buffer(quad.tobytes()), "2f 2f", "in_pos", "in_uv")])
    hud_tex = ctx.texture((HUD_W, HUD_H), 4)
    hud_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

    def render_hud_text(s):
        img = Image.new("RGBA", (HUD_W, HUD_H), (0, 0, 0, 140))
        ImageDraw.Draw(img).text((8, 8), s, font=font, fill=(120, 255, 140, 255))
        hud_tex.write(np.asarray(img).tobytes())

    render_hud_text("starting...")

    color_tex = None
    depth_tex = None
    depth_ema = None

    disp, conv, smooth = args.disp, args.conv, args.smooth
    swap, mono, paused = False, False, False
    frame = None
    seek_pending = False
    keys_prev = {}

    def edge(keysym):
        d = key_down(keysym)
        e = d and not keys_prev.get(keysym, False)
        keys_prev[keysym] = d
        return e

    def seek(secs):
        # jump by timestamp — decodes only the landing frame, not the in-between
        pos = cap.get(cv2.CAP_PROP_POS_MSEC)
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, pos + secs * 1000.0))
        depth.reset()   # clear temporal state (VDA) so it doesn't ghost across the cut

    fps_t, fps_n, fps = time.time(), 0, 0.0
    work_sum, work_n, proc_fps = 0.0, 0, 0.0
    while not glfw.window_should_close(win):
        glfw.poll_events()
        if key_down(XK.XK_q) or key_down(XK.XK_Escape):
            break
        # held tuning
        if key_down(XK.XK_minus):        disp = max(0.0, disp - 0.5)
        if key_down(XK.XK_equal):        disp += 0.5
        if key_down(XK.XK_bracketleft):  conv = max(0.0, conv - 0.02)
        if key_down(XK.XK_bracketright): conv = min(1.0, conv + 0.02)
        if key_down(XK.XK_comma):        smooth = max(0.0, smooth - 0.02)
        if key_down(XK.XK_period):       smooth = min(0.95, smooth + 0.02)
        if edge(XK.XK_s):     swap = not swap
        if edge(XK.XK_m):     mono = not mono
        if edge(XK.XK_space): paused = not paused
        if edge(XK.XK_r):     disp, conv, smooth = args.disp, args.conv, args.smooth
        if edge(XK.XK_Left):  seek(-10); seek_pending = True   # back 10s
        if edge(XK.XK_Right): seek(+10); seek_pending = True   # fwd 10s

        t_work = time.time()

        if not paused or frame is None or seek_pending:
            seek_pending = False
            ok, bgr = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)     # loop
                depth.reset()
                ok, bgr = cap.read()
                if not ok: break
            if bgr.shape[1] > args.max_w:               # downscale (e.g. 4K -> panel width)
                nh = int(bgr.shape[0] * args.max_w / bgr.shape[1])
                bgr = cv2.resize(bgr, (args.max_w, nh), interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            # depth
            d = normalize_depth(depth(frame))            # hxw GPU 0..1
            if smooth > 0 and depth_ema is not None and depth_ema.shape == d.shape:
                d = smooth * depth_ema + (1 - smooth) * d
            depth_ema = d
            dnp = d.detach().cpu().numpy().astype("f4")

            # upload textures
            if color_tex is None or color_tex.size != (frame.shape[1], frame.shape[0]):
                if color_tex: color_tex.release()
                color_tex = ctx.texture((frame.shape[1], frame.shape[0]), 3)
                color_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            color_tex.write(np.ascontiguousarray(frame))
            if depth_tex is None or depth_tex.size != (dnp.shape[1], dnp.shape[0]):
                if depth_tex: depth_tex.release()
                depth_tex = ctx.texture((dnp.shape[1], dnp.shape[0]), 1, dtype="f4")
                depth_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            depth_tex.write(np.ascontiguousarray(dnp))

        # render — viewport from the fixed panel size (pw/ph), NOT glfw's tracked
        # framebuffer (which can be stale after the override-redirect map).
        ctx.screen.use()
        ctx.clear(0.0, 0.0, 0.0)
        color_tex.use(0); depth_tex.use(1)
        disp_ndc = disp / (eye_w / 2.0)
        prog["disp"].value = 0.0 if mono else disp_ndc
        prog["conv"].value = conv
        for half_i in range(2):
            x0 = half_i * eye_w
            ctx.viewport = (x0, 0, eye_w, ph)
            left = (half_i == 0) != swap
            prog["eye_sign"].value = 1.0 if left else -1.0
            vao.render()

        # stats HUD over both eyes
        ctx.enable(moderngl.BLEND)
        hud_tex.use(0)
        for half_i in range(2):
            ctx.viewport = (half_i * eye_w, 0, eye_w, ph)
            hud_vao.render()
        ctx.disable(moderngl.BLEND)

        # processing capacity = work time WITHOUT the vsync wait (finish() forces
        # the GPU to complete so the timing is honest). display fps is vsync-capped.
        ctx.finish()
        work_sum += time.time() - t_work; work_n += 1
        glfw.swap_buffers(win)

        fps_n += 1
        if time.time() - fps_t >= 1.0:
            fps = fps_n / (time.time() - fps_t); fps_t = time.time(); fps_n = 0
            proc_fps = (work_n / work_sum) if work_sum > 0 else 0.0
            work_ms = (work_sum / work_n * 1000) if work_n else 0.0
            work_sum, work_n = 0.0, 0
            stats = (f"{model_label} | proc {proc_fps:.0f}fps ({work_ms:.0f}ms) disp {fps:.0f}  "
                     f"d{disp:.0f} c{conv:.2f} s{smooth:.2f} "
                     f"{'MONO' if mono else 'SBS'}{' SWAP' if swap else ''}")
            render_hud_text(stats)
            print("\r" + stats + "      ", end="", flush=True)

    cap.release()
    glfw.terminate()
    print()


if __name__ == "__main__":
    main()
