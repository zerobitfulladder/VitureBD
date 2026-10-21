"""model.py — a glTF / GLB / VRM model in GL, for viewer.py.

trimesh reads the file. Every mesh of its scene, with its node transform baked
in, becomes one draw: a vertex buffer (position, normal, uv, colour) and its
material. The file's own normals are kept (trimesh would recompute them on any
transform, and without scipy that takes minutes); a mesh without them gets
area-weighted ones. The model is centred on its bounding box and scaled so its
largest side is 1; draw() scales it from there.

Two kinds of material:

  PBR    glTF's metallic-roughness: base colour, metallic and roughness
         (factors and texture), normal map, emission and occlusion. GGX
         specular and Lambert diffuse under three directional lights fixed to
         the model's surroundings (key, fill, rim), plus a sky / ground ambient;
         then ACES tone mapping to sRGB.
  MToon  VRM 0.x avatars' toon shader, read from the file's VRM extension: a
         lit and a shade colour split by a hard ramp under the key light
         (shade shift and toony), a flat ambient, the colour kept at or under
         the lit colour, rim light, an additive matcap, emission, and an
         outline drawn as the mesh's back faces pushed out along the normals.
         After VRM's MToon by Masataka Sumi (MIT), as V-Sekai's
         Godot-MToon-Shader ports it. VRM 0.x avatars face -z; they are turned
         to face the viewer.

Normal maps take their tangent frame from screen-space derivatives, so no
tangents are needed. Cut-out alpha is discarded; blended parts are drawn after
the rest, in their render queue's order, farthest first.
"""
import io
import json
import struct

import moderngl
import numpy as np
import trimesh

VS = """
#version 330
in vec3 in_pos;
in vec3 in_nrm;
in vec2 in_uv;
in vec4 in_col;
uniform mat4 mvp;       // model -> clip
uniform mat4 mv;        // model -> eye
uniform mat3 nmat;      // model -> eye, for normals
out vec3 v_pos;
out vec3 v_nrm;
out vec2 v_uv;
out vec4 v_col;
void main() {
    gl_Position = mvp * vec4(in_pos, 1.0);
    v_pos = (mv * vec4(in_pos, 1.0)).xyz;
    v_nrm = nmat * in_nrm;
    v_uv = in_uv;
    v_col = in_col;
}
"""
FS = """
#version 330
in vec3 v_pos;
in vec3 v_nrm;
in vec2 v_uv;
in vec4 v_col;
uniform sampler2D base_tex;
uniform sampler2D mr_tex;
uniform sampler2D normal_tex;
uniform sampler2D emissive_tex;
uniform sampler2D occlusion_tex;
uniform int has;               // textures present: 1 base, 2 mr, 4 normal, 8 emissive, 16 occlusion
uniform vec4 base_factor;
uniform float metallic;
uniform float roughness;
uniform float normal_scale;
uniform float occlusion_strength;
uniform vec3 emissive_factor;
uniform int alpha_mode;        // 0 opaque, 1 mask, 2 blend
uniform float alpha_cutoff;
uniform vec3 light_dir[3];     // toward each light, eye space
uniform vec3 light_col[3];
uniform vec3 sky;              // ambient from above ...
uniform vec3 ground;           // ... and from below
uniform vec3 up;               // the room's up, eye space
uniform float exposure;
out vec4 f;
const float PI = 3.14159265;

vec3 linear(vec3 c) { return pow(c, vec3(2.2)); }
vec3 aces(vec3 x) {            // Narkowicz's fit of the ACES filmic curve
    return clamp(x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

void main() {
    vec4 base = base_factor * v_col;
    if ((has & 1) != 0) {
        vec4 t = texture(base_tex, v_uv);
        base *= vec4(linear(t.rgb), t.a);
    }
    if (alpha_mode == 1 && base.a < alpha_cutoff) discard;
    float m = metallic, r = roughness;
    if ((has & 2) != 0) {
        vec4 t = texture(mr_tex, v_uv);
        r *= t.g;
        m *= t.b;
    }
    r = clamp(r, 0.04, 1.0);

    vec3 n = normalize(v_nrm);
    if (!gl_FrontFacing) n = -n;
    if ((has & 4) != 0) {      // cotangent frame from derivatives (Schueler)
        vec3 dp1 = dFdx(v_pos), dp2 = dFdy(v_pos);
        vec2 du1 = dFdx(v_uv), du2 = dFdy(v_uv);
        vec3 dp2perp = cross(dp2, n), dp1perp = cross(n, dp1);
        vec3 t = dp2perp * du1.x + dp1perp * du2.x;
        vec3 b = dp2perp * du1.y + dp1perp * du2.y;
        float s = inversesqrt(max(max(dot(t, t), dot(b, b)), 1e-20));
        vec3 tn = texture(normal_tex, v_uv).xyz * 2.0 - 1.0;
        tn.xy *= normal_scale;
        n = normalize(mat3(t * s, b * s, n) * tn);
    }

    vec3 v = normalize(-v_pos);
    float nv = max(dot(n, v), 1e-4);
    vec3 f0 = mix(vec3(0.04), base.rgb, m);
    vec3 diffuse = base.rgb * (1.0 - m);
    float a = r * r, a2 = a * a, k = a * 0.5;
    vec3 col = vec3(0.0);
    for (int i = 0; i < 3; i++) {
        vec3 l = light_dir[i];
        vec3 h = normalize(l + v);
        float nl = max(dot(n, l), 0.0);
        float nh = max(dot(n, h), 0.0);
        float d = a2 / (PI * pow(nh * nh * (a2 - 1.0) + 1.0, 2.0));
        float g = nv / (nv * (1.0 - k) + k) * nl / (nl * (1.0 - k) + k);
        vec3 fr = f0 + (1.0 - f0) * pow(1.0 - max(dot(v, h), 0.0), 5.0);
        col += ((1.0 - fr) * diffuse / PI + d * g * fr / max(4.0 * nv * nl, 1e-4))
               * light_col[i] * nl;
    }
    float ao = 1.0;
    if ((has & 16) != 0) ao = mix(1.0, texture(occlusion_tex, v_uv).r, occlusion_strength);
    vec3 ambient = mix(ground, sky, 0.5 + 0.5 * dot(n, up));
    vec3 fa = f0 + (max(vec3(1.0 - r), f0) - f0) * pow(1.0 - nv, 5.0);
    col += ambient * (diffuse + 0.5 * fa) * ao;
    vec3 e = emissive_factor;
    if ((has & 8) != 0) e *= linear(texture(emissive_tex, v_uv).rgb);
    col += e;
    f = vec4(pow(aces(col * exposure), vec3(1.0 / 2.2)), alpha_mode == 2 ? base.a : 1.0);
}
"""

TOON_VS = """
#version 330
in vec3 in_pos;
in vec3 in_nrm;
in vec2 in_uv;
uniform mat4 mvp;
uniform mat4 mv;
uniform mat3 nmat;
uniform float outline;         // the outline pass: push out along the normal this far; 0 = the surface
uniform sampler2D outline_tex; // the outline's width per texel (red)
uniform int has;
out vec3 v_pos;
out vec3 v_nrm;
out vec2 v_uv;
void main() {
    vec3 p = in_pos;
    if (outline > 0.0)
        p += normalize(in_nrm) * outline * ((has & 32) != 0 ? textureLod(outline_tex, in_uv, 0.0).r : 1.0);
    gl_Position = mvp * vec4(p, 1.0);
    v_pos = (mv * vec4(p, 1.0)).xyz;
    v_nrm = nmat * in_nrm;
    v_uv = in_uv;
}
"""
TOON_FS = """
#version 330
in vec3 v_pos;
in vec3 v_nrm;
in vec2 v_uv;
uniform sampler2D main_tex;
uniform sampler2D shade_tex;
uniform sampler2D bump_tex;
uniform sampler2D sphere_tex;
uniform sampler2D emission_tex;
uniform int has;               // textures present: 1 main, 2 shade, 4 bump, 8 matcap, 16 emission, 32 outline width
uniform vec4 color;            // the lit colour (linear)
uniform vec3 shade_color;
uniform float shade_toony;
uniform float shade_shift;
uniform float bump_scale;
uniform float indirect;        // how much the ambient follows the normal
uniform float light_atten;     // how much light colour is ignored (to its brightest channel)
uniform vec3 rim_color;
uniform float rim_power;
uniform float rim_lift;
uniform float rim_mix;
uniform vec3 emission_color;
uniform int blend;             // 0 opaque, 1 cut out, 2 / 3 blended
uniform float cutoff;
uniform float outline;         // > 0: the outline pass
uniform int outline_mixed;     // its colour: 0 fixed, 1 mixed with the shading
uniform vec3 outline_color;
uniform float outline_mix;
uniform vec3 light_dir;        // toward the key light, eye space
uniform vec3 light_col;
uniform vec3 sky;
uniform vec3 ground;
uniform vec3 up;
out vec4 f;

vec3 linear(vec3 c) { return pow(c, vec3(2.2)); }
float max3(vec3 c) { return max(c.x, max(c.y, c.z)); }

void main() {
    vec4 t = (has & 1) != 0 ? texture(main_tex, v_uv) : vec4(1.0);
    vec4 lit = color * vec4(linear(t.rgb), t.a);
    if (blend == 1 && lit.a < cutoff) discard;
    float alpha = blend >= 2 ? lit.a : 1.0;
    vec3 shade = shade_color * ((has & 2) != 0 ? linear(texture(shade_tex, v_uv).rgb) : vec3(1.0));

    vec3 n = normalize(v_nrm);
    if (!gl_FrontFacing) n = -n;
    if ((has & 4) != 0) {      // cotangent frame from derivatives (Schueler)
        vec3 dp1 = dFdx(v_pos), dp2 = dFdy(v_pos);
        vec2 du1 = dFdx(v_uv), du2 = dFdy(v_uv);
        vec3 dp2perp = cross(dp2, n), dp1perp = cross(n, dp1);
        vec3 tg = dp2perp * du1.x + dp1perp * du2.x;
        vec3 bt = dp2perp * du1.y + dp1perp * du2.y;
        float s = inversesqrt(max(max(dot(tg, tg), dot(bt, bt)), 1e-20));
        vec3 tn = texture(bump_tex, v_uv).xyz * 2.0 - 1.0;
        tn.xy *= bump_scale;
        n = normalize(mat3(tg * s, bt * s, n) * tn);
    }
    vec3 v = normalize(-v_pos);

    // the toon ramp: shade below shade_shift, lit above, the step's width by toony
    float hi = mix(1.0, shade_shift, shade_toony);
    float li = clamp((dot(n, light_dir) - shade_shift) / max(hi - shade_shift, 1e-5), 0.0, 1.0);
    vec3 lighting = mix(light_col, vec3(max(max3(light_col), 1e-5)), light_atten);
    vec3 col = mix(shade, lit.rgb, li) * lighting;
    // a flat ambient, toward the normal's own by `indirect`; never brighter than lit
    vec3 gi = mix(0.5 * (sky + ground), mix(ground, sky, 0.5 + 0.5 * dot(n, up)), indirect);
    gi = mix(gi, vec3(max3(gi)), light_atten);
    col = min(col + gi * lit.rgb, lit.rgb);
    // rim, matcap, emission
    col += pow(clamp(1.0 - dot(n, v) + rim_lift, 0.0, 1.0), max(rim_power, 1e-3)) * rim_color
           * mix(vec3(1.0), lighting + gi, rim_mix);
    if ((has & 8) != 0) {
        vec3 vup = normalize(vec3(0.0, 1.0, 0.0) - v * v.y);
        vec3 vright = normalize(cross(vup, v));
        col += linear(texture(sphere_tex, vec2(dot(vright, n), dot(vup, n)) * 0.5 + 0.5).rgb);
    }
    col += emission_color * ((has & 16) != 0 ? linear(texture(emission_tex, v_uv).rgb) : vec3(1.0));
    if (outline > 0.0)
        col = outline_mixed == 1 ? outline_color * mix(vec3(1.0), col, outline_mix) : outline_color;
    f = vec4(pow(max(col, vec3(0.0)), vec3(1.0 / 2.2)), alpha);
}
"""
# MToon's textures, by texture unit, and their bit in `has`
TOON_TEXTURES = ("_MainTex", "_ShadeTexture", "_BumpMap", "_SphereAdd", "_EmissionMap",
                 "_OutlineWidthTexture")

# The lights, in the model's surroundings (x right, y up, z toward the viewer
# when it was placed): a warm key from the upper left, a cool fill from the
# right, a rim from behind; and the ambient.
LIGHTS = (((-0.5, 0.8, 0.6), (2.6, 2.45, 2.25)),
          ((0.7, 0.2, 0.5), (0.7, 0.8, 0.95)),
          ((0.1, 0.5, -1.0), (1.4, 1.4, 1.5)))
SKY, GROUND = (0.32, 0.35, 0.42), (0.12, 0.10, 0.09)
# MToon takes the key light, white, and its own ambient: toon shading reads the
# light's direction, and keeps colours at the material's own
TOON_LIGHT, TOON_SKY, TOON_GROUND = (1.0, 1.0, 1.0), (0.55, 0.57, 0.62), (0.32, 0.30, 0.29)


def _glb(path):
    """A binary glTF's (.glb, .vrm) JSON and binary chunk, or (None, None)."""
    with open(path, "rb") as f:
        head = f.read(12)
        if len(head) < 12 or struct.unpack("<I", head[:4])[0] != 0x46546C67:
            return None, None
        data = f.read()
    n = struct.unpack_from("<I", data, 0)[0]
    gltf = json.loads(data[8:8 + n])
    rest = data[8 + n:]
    blob = rest[8:8 + struct.unpack_from("<I", rest, 0)[0]] if len(rest) >= 8 else b""
    return gltf, blob


def _srgb(v):
    """A colour as stored by VRM (sRGB, 0..1) to linear."""
    return tuple(float(max(c, 0.0)) ** 2.2 for c in v[:3])


def _vertex_normals(pos, faces):
    """Area-weighted vertex normals, for meshes that come without any."""
    fn = np.cross(pos[faces[:, 1]] - pos[faces[:, 0]], pos[faces[:, 2]] - pos[faces[:, 0]])
    vn = np.zeros_like(pos)
    for i in range(3):
        np.add.at(vn, faces[:, i], fn)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)


def _meshes(scene):
    """Each mesh of a loaded file as (mesh, positions, normals), positions and
    normals in the scene's frame."""
    if isinstance(scene, trimesh.Trimesh):
        nodes = [(np.eye(4), scene)]
    else:
        nodes = [(scene.graph[node][0], scene.geometry[scene.graph[node][1]])
                 for node in scene.graph.nodes_geometry]
    out = []
    for t, g in nodes:
        if not isinstance(g, trimesh.Trimesh) or not len(g.faces):
            continue
        pos = np.asarray(g.vertices, np.float64)
        faces = np.asarray(g.faces)
        nrm = (np.asarray(g.vertex_normals, np.float64) if "vertex_normals" in g._cache.cache
               else _vertex_normals(pos, faces))
        pos = pos @ t[:3, :3].T + t[:3, 3]
        nrm = nrm @ np.linalg.inv(t[:3, :3])                # by the inverse transpose
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
        out.append((g, pos, nrm))
    return out


def _rgba(v, n):
    """A colour factor as 4 floats in 0..1 (trimesh keeps some as 0..255)."""
    v = np.ones(4) if v is None else np.asarray(v, dtype=np.float64).ravel()
    if v.size < n:
        v = np.concatenate([v, np.ones(n - v.size)])
    return v / 255.0 if v.max() > 1.001 else v


class Model:
    def __init__(self, ctx: moderngl.Context, path: str):
        """Load `path` (.glb, .gltf, .vrm) into GL. Raises ValueError if it holds no
        triangles."""
        self.ctx = ctx
        gltf, blob = _glb(path)
        vrm = (gltf or {}).get("extensions", {}).get("VRM")
        self._gltf, self._blob = gltf, blob
        toon = {m["name"]: m for m in vrm["materialProperties"]
                if m.get("shader") == "VRM/MToon"} if vrm else {}
        meshes = _meshes(trimesh.load(path, file_type="glb" if gltf else None, process=False))
        if not meshes:
            raise ValueError("no triangles in it")

        lo = np.min([pos.min(axis=0) for _, pos, _ in meshes], axis=0)
        hi = np.max([pos.max(axis=0) for _, pos, _ in meshes], axis=0)
        centre, self.unit = (lo + hi) / 2, 1.0 / max(float(np.max(hi - lo)), 1e-12)
        self.extent = (hi - lo) * self.unit          # the model's size, largest side 1
        turn = np.array([-1.0, 1.0, -1.0]) if vrm else np.ones(3)   # VRM 0.x faces -z

        self.pbr = ctx.program(vertex_shader=VS, fragment_shader=FS)
        for i, name in enumerate(("base_tex", "mr_tex", "normal_tex", "emissive_tex",
                                  "occlusion_tex")):
            self.pbr[name] = i
        self.toon = None
        if toon:
            self.toon = ctx.program(vertex_shader=TOON_VS, fragment_shader=TOON_FS)
            for i, name in enumerate(("main_tex", "shade_tex", "bump_tex", "sphere_tex",
                                      "emission_tex", "outline_tex")):
                self.toon[name] = i
        self._textures: dict[int, moderngl.Texture] = {}
        self._images: dict[int, object] = {}
        self.parts = []
        self.triangles = 0
        for m, pos, nrm in meshes:
            pos, nrm = (pos - centre) * self.unit * turn, nrm * turn
            name = getattr(getattr(m.visual, "material", None), "name", None)
            if name in toon:
                self.parts.append(self._toon_part(m, pos, nrm, toon[name]))
            else:
                self.parts.append(self._pbr_part(m, pos, nrm))
            self.triangles += len(m.faces)

    # --- loading ---------------------------------------------------------------

    def _texture(self, image):
        if image is None:
            return None
        t = self._textures.get(id(image))
        if t is None:
            rgba = np.ascontiguousarray(np.asarray(image.convert("RGBA"), np.uint8)[::-1])
            t = self.ctx.texture((rgba.shape[1], rgba.shape[0]), 4, rgba.tobytes())
            t.build_mipmaps()
            t.anisotropy = 8.0
            self._textures[id(image)] = t
        return t

    def _image(self, index: int):
        """The image of the glTF texture `index`, from the binary chunk."""
        if index not in self._images:
            from PIL import Image
            g = self._gltf
            view = g["bufferViews"][g["images"][g["textures"][index]["source"]]["bufferView"]]
            start = view.get("byteOffset", 0)
            self._images[index] = Image.open(io.BytesIO(self._blob[start:start + view["byteLength"]]))
        return self._images[index]

    def _buffers(self, m, pos, nrm, prog, fmt):
        n = len(pos)
        visual = m.visual
        uv = getattr(visual, "uv", None)
        uv = np.zeros((n, 2)) if uv is None or len(uv) != n else np.asarray(uv)[:, :2]
        col = np.ones((n, 4))
        if isinstance(visual, trimesh.visual.ColorVisuals) and visual.kind == "vertex":
            col = np.asarray(visual.vertex_colors, np.float64) / 255.0
        data = np.hstack([pos, nrm, uv, col]).astype("f4")
        vbo = self.ctx.buffer(np.ascontiguousarray(data).tobytes())
        ibo = self.ctx.buffer(np.ascontiguousarray(m.faces, np.uint32).tobytes())
        names = ("in_pos", "in_nrm", "in_uv", "in_col")[:len(fmt.split()) - fmt.count("x")]
        return self.ctx.vertex_array(prog, [(vbo, fmt, *names)], ibo), (vbo, ibo)

    def _pbr_part(self, m, pos, nrm):
        vao, buffers = self._buffers(m, pos, nrm, self.pbr, "3f 3f 2f 4f")
        mat = getattr(m.visual, "material", None)
        if mat is not None and not isinstance(mat, trimesh.visual.material.PBRMaterial):
            mat = mat.to_pbr()
        get = (lambda k, d=None: getattr(mat, k, d) if mat is not None else d)
        tex = [self._texture(get("baseColorTexture")),
               self._texture(get("metallicRoughnessTexture")),
               self._texture(get("normalTexture")),
               self._texture(get("emissiveTexture")),
               self._texture(get("occlusionTexture"))]
        emissive = get("emissiveFactor")
        alpha = {"MASK": 1, "BLEND": 2}.get(get("alphaMode") or "OPAQUE", 0)
        return {
            "kind": "pbr", "vao": vao, "buffers": buffers, "tex": tex,
            "blend": alpha == 2, "zwrite": alpha < 2, "queue": (2000, 2450, 3000)[alpha],
            "cull": None if get("doubleSided", False) else "back", "outline": 0.0,
            "centre": (pos.min(axis=0) + pos.max(axis=0)) / 2,
            "uniforms": {
                "has": sum(1 << i for i, t in enumerate(tex) if t is not None),
                "base_factor": tuple(_rgba(get("baseColorFactor"), 4)),
                "metallic": float(1.0 if get("metallicFactor") is None else get("metallicFactor")),
                "roughness": float(1.0 if get("roughnessFactor") is None else get("roughnessFactor")),
                "emissive_factor": tuple((np.zeros(3) if emissive is None else _rgba(emissive, 3))[:3]),
                "alpha_mode": alpha, "alpha_cutoff": float(get("alphaCutoff") or 0.5),
            },
        }

    def _toon_part(self, m, pos, nrm, props):
        vao, buffers = self._buffers(m, pos, nrm, self.toon, "3f 3f 2f 16x")
        fl, vec, texs = props["floatProperties"], props["vectorProperties"], props["textureProperties"]
        tex = [self._texture(self._image(texs[k])) if k in texs else None for k in TOON_TEXTURES]
        blend = int(fl.get("_BlendMode", 0))
        cull = {0: None, 1: "front"}.get(int(fl.get("_CullMode", 2)), "back")
        width = fl.get("_OutlineWidth", 0.0) if fl.get("_OutlineWidthMode", 0) >= 1 else 0.0
        color = vec.get("_Color", [1, 1, 1, 1])
        return {
            "kind": "toon", "vao": vao, "buffers": buffers, "tex": tex,
            "blend": blend >= 2, "zwrite": blend != 2, "queue": props.get("renderQueue", 2000),
            "cull": cull, "outline": 0.01 * width * self.unit,     # cm in the file
            "centre": (pos.min(axis=0) + pos.max(axis=0)) / 2,
            "uniforms": {
                "has": sum(1 << i for i, t in enumerate(tex) if t is not None),
                "color": (*_srgb(color), float(color[3])),
                "shade_color": _srgb(vec.get("_ShadeColor", [1, 1, 1])),
                "shade_toony": float(fl.get("_ShadeToony", 0.9)),
                "shade_shift": float(fl.get("_ShadeShift", 0.0)),
                "bump_scale": float(fl.get("_BumpScale", 1.0)),
                "indirect": float(fl.get("_IndirectLightIntensity", 0.1)),
                "light_atten": float(fl.get("_LightColorAttenuation", 0.0)),
                "rim_color": _srgb(vec.get("_RimColor", [0, 0, 0])),
                "rim_power": float(fl.get("_RimFresnelPower", 1.0)),
                "rim_lift": float(fl.get("_RimLift", 0.0)),
                "rim_mix": float(fl.get("_RimLightingMix", 0.0)),
                "emission_color": _srgb(vec.get("_EmissionColor", [0, 0, 0])),
                "blend": blend, "cutoff": float(fl.get("_Cutoff", 0.5)),
                "outline_mixed": int(fl.get("_OutlineColorMode", 0)),
                "outline_color": _srgb(vec.get("_OutlineColor", [0, 0, 0])),
                "outline_mix": float(fl.get("_OutlineLightingMix", 1.0)),
            },
        }

    # --- drawing ---------------------------------------------------------------

    def draw(self, proj: np.ndarray, mv: np.ndarray, room: np.ndarray, exposure: float) -> None:
        """Draw into the current viewport (depth-tested): `mv` takes model space
        (as loaded: centred, largest side 1) to eye space, `room` (3x3) turns the
        model's surroundings into eye space, for the lights."""
        ctx = self.ctx
        up = tuple(room @ (0.0, 1.0, 0.0))
        dirs = [room @ np.asarray(d, np.float64) for d, _ in LIGHTS]
        dirs = [d / np.linalg.norm(d) for d in dirs]
        for prog in (self.pbr, self.toon):
            if prog is None:
                continue
            _set(prog, "mvp", (proj @ mv).T.astype("f4").tobytes())
            _set(prog, "mv", mv.T.astype("f4").tobytes())
            _set(prog, "nmat", np.linalg.inv(mv[:3, :3]).astype("f4").tobytes())   # (inverse)^T, column-major
            _set(prog, "up", up)
        p = self.pbr
        _set(p, "light_dir", np.array(dirs, "f4").tobytes())
        _set(p, "light_col", np.array([c for _, c in LIGHTS], "f4").tobytes())
        _set(p, "sky", SKY)
        _set(p, "ground", GROUND)
        _set(p, "exposure", exposure)
        _set(p, "normal_scale", 1.0)
        _set(p, "occlusion_strength", 1.0)
        if self.toon is not None:
            t = self.toon
            _set(t, "light_dir", tuple(dirs[0]))
            _set(t, "light_col", tuple(c * exposure for c in TOON_LIGHT))
            _set(t, "sky", tuple(c * exposure for c in TOON_SKY))
            _set(t, "ground", tuple(c * exposure for c in TOON_GROUND))

        opaque = [p for p in self.parts if not p["blend"]]
        blend = sorted((p for p in self.parts if p["blend"]),
                       key=lambda p: (p["queue"], (mv @ np.append(p["centre"], 1.0))[2]))
        ctx.enable(moderngl.DEPTH_TEST)
        for p in opaque:
            self._draw_part(p)
        if blend:
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            for p in blend:
                ctx.depth_mask = p["zwrite"]
                self._draw_part(p)
            ctx.depth_mask = True
            ctx.disable(moderngl.BLEND)
        ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

    def _draw_part(self, p) -> None:
        ctx = self.ctx
        prog = p["vao"].program
        for i, t in enumerate(p["tex"]):
            if t is not None:
                t.use(i)
        for k, v in p["uniforms"].items():
            _set(prog, k, v)
        _cull(ctx, p["cull"])
        _set(prog, "outline", 0.0)
        p["vao"].render(moderngl.TRIANGLES)
        if p["outline"] > 0.0:              # the outline: the pushed-out back faces
            _cull(ctx, "front")
            _set(prog, "outline", p["outline"])
            p["vao"].render(moderngl.TRIANGLES)

    def release(self) -> None:
        for p in self.parts:
            p["vao"].release()
            for b in p["buffers"]:
                b.release()
        for t in self._textures.values():
            t.release()
        for prog in (self.pbr, self.toon):
            if prog is not None:
                prog.release()


def _set(prog, name, value) -> None:
    """Set a uniform the compiler kept (an unused one is optimized out)."""
    u = prog.get(name, None)
    if u is None:
        return
    if isinstance(value, bytes):
        u.write(value)
    else:
        u.value = value


def _cull(ctx, face) -> None:
    """Cull "front" or "back" faces, or (None) neither."""
    if face is None:
        ctx.disable(moderngl.CULL_FACE)
    else:
        ctx.enable(moderngl.CULL_FACE)
        ctx.cull_face = face
