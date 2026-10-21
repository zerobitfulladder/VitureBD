"""splats.py — a 3D Gaussian splat scene in GL, for viewer.py.

Each splat is a 3D Gaussian: a centre, a covariance (from its scale and
rotation), a colour and an opacity. They are kept in a float texture, four
texels each. Every frame the splats are sorted by depth on the GPU (torch),
and the order goes straight into a GL buffer (glcuda.GLBufferWriter). Each
splat is then drawn as a quad around its centre, shaped by its covariance
projected onto the screen (Kerbl et al.'s EWA splatting, as antimatter15's and
gsplat.js's WebGL renderers draw it), and blended back to front onto black.

Formats: antimatter15's .splat (32 bytes each: position, scale, RGBA,
rotation) and the 3DGS .ply. A .ply's colour is its zeroth spherical
harmonic; the higher bands, the view-dependent part, are dropped, as .splat
drops them. Both come in COLMAP's axes (y down, the subject seen along +z)
and are turned to y up, facing the viewer.

A capture holds its surroundings too, as splats spread far and wide, and all
of it is drawn; splats behind the eye are skipped. So that the subject, not
the middle of the room, is where the scene is placed and what SIZE measures,
the subject is found as the scene's dense cluster (as zerobitfulladder's
ply2splat.py finds it): a weighted centre, taken again over the nearer half of
the splats until it settles. The scene is centred there and scaled so the
subject (90% of that half) spans 1, as model.Model's largest side does.
"""
import moderngl
import numpy as np

SPLAT_TYPES = (".splat", ".ply")
WIDTH = 4096                    # the data texture's width: 1024 splats a row, 4 texels each
SH_C0 = 0.28209479177387814     # the zeroth spherical harmonic

VS = """
#version 330
in vec2 in_corner;              // the quad's corner, in sqrt(2) standard deviations
in uint in_index;               // which splat (per instance, in depth order)
uniform sampler2D data;         // per splat: centre, covariance (2 texels), colour
uniform mat4 mv;                // model -> eye
uniform mat4 proj;
uniform vec2 viewport;          // px
out vec4 v_col;
out vec2 v_pos;
void main() {
    int i = int(in_index) * 4;
    ivec2 at = ivec2(i % TEXW, i / TEXW);
    vec4 c0 = texelFetch(data, at, 0);
    vec4 c1 = texelFetch(data, at + ivec2(1, 0), 0);
    vec4 c2 = texelFetch(data, at + ivec2(2, 0), 0);
    vec4 cam = mv * vec4(c0.xyz, 1.0);
    vec4 clip = proj * cam;
    float bound = 1.2 * clip.w;
    if (cam.z > -0.01 || abs(clip.x) > bound || abs(clip.y) > bound) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);    // behind the eye or off screen
        return;
    }
    mat3 sigma = mat3(c1.x, c1.y, c1.z, c1.y, c1.w, c2.x, c1.z, c2.x, c2.y);
    // the screen's Jacobian at the centre (px per eye-space metre)
    float z = -cam.z;
    float fx = proj[0][0] * viewport.x * 0.5, fy = proj[1][1] * viewport.y * 0.5;
    mat3 j = mat3(fx / z, 0.0, 0.0,
                  0.0, fy / z, 0.0,
                  fx * cam.x / (z * z), fy * cam.y / (z * z), 0.0);
    mat3 t = j * mat3(mv);
    mat3 cov = t * sigma * transpose(t);
    float a = cov[0][0] + 0.3, b = cov[0][1], c = cov[1][1] + 0.3;   // at least ~a pixel wide
    float mid = 0.5 * (a + c), r = length(vec2(0.5 * (a - c), b));
    float l1 = mid + r, l2 = mid - r;
    if (l2 < 0.0) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        return;
    }
    vec2 d = normalize(vec2(b, l1 - a));
    vec2 major = min(sqrt(2.0 * l1), 1024.0) * d;
    vec2 minor = min(sqrt(2.0 * l2), 1024.0) * vec2(d.y, -d.x);
    gl_Position = vec4(clip.xy / clip.w
                       + (in_corner.x * major + in_corner.y * minor) * 2.0 / viewport, 0.0, 1.0);
    v_col = texelFetch(data, at + ivec2(3, 0), 0);
    v_pos = in_corner;
}
""".replace("TEXW", str(WIDTH))
FS = """
#version 330
in vec4 v_col;
in vec2 v_pos;
uniform float exposure;
out vec4 f;
void main() {
    float e = -dot(v_pos, v_pos);
    if (e < -4.0) discard;
    float alpha = exp(e) * v_col.a;
    f = vec4(v_col.rgb * exposure * alpha, alpha);  // premultiplied, blended over what is behind
}
"""


def _subject(pos, weight):
    """The dense cluster's centre: a weighted mean, re-taken over the nearer
    half of the splats until it settles."""
    w = weight + 1e-6
    centre = np.average(pos, axis=0, weights=w)
    for _ in range(20):
        d = np.linalg.norm(pos - centre, axis=1)
        inside = d <= np.median(d)
        moved = np.average(pos[inside], axis=0, weights=w[inside])
        if np.linalg.norm(moved - centre) < 1e-4 * np.median(d):
            return moved
        centre = moved
    return centre


def is_splats(path: str) -> bool:
    return path.lower().endswith(SPLAT_TYPES)


def _read_splat(path):
    """antimatter15's .splat: position, scale, RGBA and rotation (w first)."""
    raw = np.fromfile(path, dtype=np.dtype([("pos", "<f4", 3), ("scale", "<f4", 3),
                                            ("rgba", "u1", 4), ("rot", "u1", 4)]))
    return (raw["pos"].astype(np.float64), raw["scale"].astype(np.float64),
            raw["rgba"] / 255.0, (raw["rot"].astype(np.float64) - 128.0) / 128.0)


def _read_ply(path):
    """A 3DGS .ply: its properties by name, from the header."""
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError("not a PLY file")
        fields, count, fmt = [], 0, None
        while True:
            line = f.readline().decode("ascii", "replace").split()
            if not line or line[0] == "end_header":
                break
            if line[0] == "format":
                fmt = line[1]
            elif line[:2] == ["element", "vertex"]:
                count = int(line[2])
            elif line[0] == "property" and count and line[1] in ("float", "float32", "double"):
                fields.append((line[2], "<f4" if line[1] != "double" else "<f8"))
        if fmt != "binary_little_endian":
            raise ValueError(f"PLY format {fmt}: only binary_little_endian is read")
        v = np.fromfile(f, dtype=np.dtype(fields), count=count)
    names = set(v.dtype.names)
    if not {"f_dc_0", "opacity", "scale_0", "rot_0"} <= names:
        raise ValueError("not a Gaussian splat PLY (no f_dc / opacity / scale / rot)")
    col = 0.5 + SH_C0 * np.stack([v[f"f_dc_{i}"] for i in range(3)], 1)
    alpha = 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float64)))
    return (np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64),
            np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], 1).astype(np.float64)),
            np.clip(np.concatenate([col, alpha[:, None]], 1), 0.0, 1.0),
            np.stack([v[f"rot_{i}"] for i in range(4)], 1).astype(np.float64))


def _covariances(scale, rot):
    """Each splat's 3D covariance, R S S^T R^T, from its scale and rotation."""
    q = rot / np.maximum(np.linalg.norm(rot, axis=1, keepdims=True), 1e-12)
    w, x, y, z = q.T
    r = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                  2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                  2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1)
    m = r.reshape(-1, 3, 3) * scale[:, None, :]
    return m @ m.transpose(0, 2, 1)


class Splats:
    def __init__(self, ctx: moderngl.Context, path: str):
        """Load `path` (.splat, or a 3DGS .ply) into GL. Raises ValueError if it
        holds no splats."""
        self.ctx = ctx
        pos, scale, rgba, rot = (_read_ply if path.lower().endswith(".ply") else _read_splat)(path)
        if not len(pos):
            raise ValueError("no splats in it")
        cov = _covariances(scale, rot)
        flip = np.array([1.0, -1.0, -1.0])                 # COLMAP's y down, +z away -> GL
        pos = pos * flip
        cov = cov * flip[:, None] * flip[None, :]
        centre = _subject(pos, rgba[:, 3])
        dist = np.linalg.norm(pos - centre, axis=1)
        near = dist[dist <= np.median(dist)]
        self.unit = 1.0 / max(2.0 * float(np.percentile(near, 90)), 1e-12)   # subject radius 0.5
        pos = (pos - centre) * self.unit
        cov = cov * self.unit ** 2
        self.count = n = len(pos)

        rows = -(-n * 4 // WIDTH)
        data = np.zeros((rows * WIDTH // 4, 4, 4), np.float32)
        data[:n, 0, :3] = pos
        data[:n, 0, 3] = 1.0
        data[:n, 1] = np.stack([cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2], cov[:, 1, 1]], 1)
        data[:n, 2, :2] = np.stack([cov[:, 1, 2], cov[:, 2, 2]], 1)
        data[:n, 3] = rgba
        self.data = ctx.texture((WIDTH, rows), 4, data.tobytes(), dtype="f4")
        self.data.filter = (moderngl.NEAREST, moderngl.NEAREST)

        self.prog = ctx.program(vertex_shader=VS, fragment_shader=FS)
        self.prog["data"] = 0
        corners = np.array([-2, -2, 2, -2, -2, 2, 2, 2], dtype="f4")
        self.quad = ctx.buffer(corners.tobytes())
        self.order = ctx.buffer(np.arange(n, dtype=np.uint32).tobytes())
        self.vao = ctx.vertex_array(self.prog, [(self.quad, "2f", "in_corner"),
                                                (self.order, "1u/i", "in_index")])
        import torch                  # here, not at import: the CLI imports this module
        from .glcuda import GLBufferWriter
        self._torch = torch
        self._writer = GLBufferWriter(self.order.glo)
        self._pos = torch.from_numpy(pos.astype(np.float32)).cuda()

    def sort(self, mv: np.ndarray) -> None:
        """Order the splats back to front for the eye `mv` (model -> eye) sees
        them from; both eyes share it."""
        torch = self._torch
        row = torch.from_numpy(mv[2, :3].astype(np.float32)).cuda()
        order = torch.argsort(self._pos @ row).to(torch.int32)   # eye z, farthest first
        self._writer.write(order)

    def draw(self, proj: np.ndarray, mv: np.ndarray, room: np.ndarray, exposure: float) -> None:
        """Draw into the current viewport, after sort(): `mv` takes model space
        (centred, largest side 1) to eye space. `room` is unused: splats carry
        their light in their colours."""
        ctx, prog = self.ctx, self.prog
        prog["mv"].write(mv.T.astype("f4").tobytes())
        prog["proj"].write(proj.T.astype("f4").tobytes())
        prog["viewport"].value = tuple(float(v) for v in ctx.viewport[2:])
        prog["exposure"].value = exposure
        self.data.use(0)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = (moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)
        self.vao.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=self.count)
        ctx.disable(moderngl.BLEND)
        ctx.blend_func = moderngl.DEFAULT_BLENDING

    def release(self) -> None:
        self._writer.close()
        for o in (self.vao, self.prog, self.quad, self.order, self.data):
            o.release()
