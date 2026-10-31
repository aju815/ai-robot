"""OpenGL hologram renderer (moderngl). Real GPU rendering of the skinned, textured humanoid model with
a holographic shader: depth pre-pass, Fresnel rim light, scan lines, light sweep, flicker, glitch bands,
region highlights, floor reflection, projector base and light cone, anti-aliased with MSAA."""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .mathutil import rot_y, scale_m, translate
from .scene import MESH_BUILDERS, RenderState

try:
    import moderngl
except Exception as _e:            # pragma: no cover
    moderngl = None
    _IMPORT_ERROR = _e


class RendererError(RuntimeError):
    pass


HOLO_VS = """
#version 330
uniform mat4 u_vp;
uniform mat4 u_model;
uniform float u_time;
uniform float u_glitch;
uniform float u_wire;
in vec3 in_pos;
in vec3 in_nor;
in vec2 in_uv;
in float in_region;
out vec3 v_world;
out vec3 v_nor;
out vec2 v_uv;
out float v_region;
invariant gl_Position;
void main() {
    vec4 w = u_model * vec4(in_pos, 1.0);
    float row = floor(w.y * 26.0 + u_time * 4.0);
    float band = step(0.9935, fract(sin(row * 127.1) * 43758.5453));
    w.x += band * u_glitch * 0.05 * sin(u_time * 55.0 + row);
    v_world = w.xyz;
    v_nor = mat3(u_model) * in_nor;
    v_uv = in_uv;
    v_region = in_region;
    gl_Position = u_vp * w;
    gl_Position.z -= u_wire * 0.0006 * gl_Position.w;
}
"""

HOLO_FS = """
#version 330
uniform vec3 u_cam;
uniform vec3 u_tint;
uniform vec3 u_tint2;
uniform float u_time;
uniform float u_alpha;
uniform int u_mode;            // 0 solid, 1 wire, 2 depth only
uniform sampler2D u_tex;
uniform int u_has_tex;
uniform vec4 u_base;
uniform float u_cutoff;        // <0: no alpha test
uniform float u_texmix;
uniform float u_hl_a;
uniform float u_hl_b;
uniform float u_hl_a_amt;
uniform float u_hl_b_amt;
uniform vec3 u_hl_col;
uniform float u_fade;
uniform float u_reflect;
in vec3 v_world;
in vec3 v_nor;
in vec2 v_uv;
in float v_region;
out vec4 f_color;
void main() {
    vec4 tex = (u_has_tex == 1) ? texture(u_tex, v_uv) : vec4(1.0);
    float a0 = tex.a * u_base.a;
    if (u_cutoff >= 0.0 && a0 < u_cutoff) discard;
    if (u_mode == 2) { f_color = vec4(0.0); return; }
    vec3 N = normalize(v_nor);
    if (!gl_FrontFacing) N = -N;
    vec3 V = normalize(u_cam - v_world);
    float ndv = clamp(abs(dot(N, V)), 0.0, 1.0);
    float fres = pow(1.0 - ndv, 2.2);
    vec3 albedo = u_base.rgb * tex.rgb;
    float lum = dot(albedo, vec3(0.299, 0.587, 0.114));
    vec3 L1 = normalize(vec3(0.45, 0.75, 0.55));
    vec3 L2 = normalize(vec3(-0.6, 0.2, 0.4));
    float diff = 0.30 + 0.55 * max(dot(N, L1), 0.0) + 0.25 * max(dot(N, L2), 0.0);
    vec3 holo = u_tint * (0.30 + 1.05 * lum) * diff;
    vec3 detail = u_tint * (albedo * 1.8) * diff;
    vec3 col = mix(holo, detail, u_texmix) + u_tint2 * fres * 1.25;
    float scan = 0.86 + 0.14 * sin(v_world.y * 210.0 - u_time * 6.0);
    float coarse = 0.93 + 0.07 * sin(v_world.y * 22.0 + u_time * 1.7);
    float sweep_y = fract(u_time * 0.22) * 2.3 - 0.15;
    float sweep = exp(-abs(v_world.y - sweep_y) * 38.0) * 0.55;
    float flick = 0.95 + 0.05 * sin(u_time * 41.0) * sin(u_time * 7.3);
    col = col * scan * coarse * flick + u_tint2 * sweep;
    float ha = 1.0 - smoothstep(0.0, 0.6, abs(v_region - u_hl_a));
    float hb = 1.0 - smoothstep(0.0, 0.6, abs(v_region - u_hl_b));
    float pulse = 0.65 + 0.35 * sin(u_time * 6.0);
    col += u_hl_col * (ha * u_hl_a_amt * 0.55 + hb * u_hl_b_amt * 0.55 * pulse);
    float alpha = u_alpha * (0.68 + 0.32 * fres);
    if (u_fade > 0.5) alpha *= smoothstep(0.0, 0.10, v_world.y);
    if (u_reflect > 0.0) alpha *= u_reflect * smoothstep(-1.2, 0.0, v_world.y);
    if (u_mode == 1) { col = u_tint2 * (0.8 + 0.8 * fres); alpha *= 0.9; }
    f_color = vec4(col, alpha);
}
"""

FLAT_VS = """
#version 330
uniform mat4 u_vp;
uniform mat4 u_model;
in vec3 in_pos;
in vec4 in_col;
out vec4 v_col;
void main() {
    v_col = in_col;
    gl_Position = u_vp * u_model * vec4(in_pos, 1.0);
}
"""
FLAT_FS = """
#version 330
uniform float u_alpha;
uniform vec3 u_tint;
in vec4 v_col;
out vec4 f_color;
void main() { f_color = vec4(v_col.rgb * u_tint, v_col.a * u_alpha); }
"""

PT_VS = """
#version 330
uniform mat4 u_vp;
uniform float u_scale;
in vec4 in_p;
void main() {
    vec4 c = u_vp * vec4(in_p.xyz, 1.0);
    gl_Position = c;
    gl_PointSize = clamp(in_p.w * u_scale / max(c.w, 0.2), 1.0, 6.0);
}
"""
PT_FS = """
#version 330
uniform vec3 u_col;
uniform float u_alpha;
out vec4 f_color;
void main() {
    float r = length(gl_PointCoord - vec2(0.5));
    float a = smoothstep(0.5, 0.0, r);
    f_color = vec4(u_col, a * u_alpha);
}
"""


def make_context(headless=False):
    """Create an OpenGL 3.3+ context. Tries the platform default, then EGL (headless / servers)."""
    if moderngl is None:
        raise RendererError(f"moderngl is not installed ({_IMPORT_ERROR}). pip install moderngl")
    errors = []
    attempts = [("egl",), ()] if headless else [(), ("egl",)]
    for a in attempts:
        try:
            if a:
                return moderngl.create_standalone_context(require=330, backend=a[0])
            return moderngl.create_standalone_context(require=330)
        except Exception as e:                       # noqa: BLE001
            errors.append(f"{a[0] if a else 'default'}: {e}")
    raise RendererError("could not create an OpenGL 3.3 context -> " + " | ".join(errors))


class _GLPrim:
    pass


class GLRenderer:
    name = "OpenGL"

    def __init__(self, width: int, height: int, samples: int = 4, wire: bool = True,
                 headless: bool = False):
        self.w, self.h = int(width), int(height)
        self.wire = wire
        self.ctx = make_context(headless)
        ctx = self.ctx
        self.info = ctx.info.get("GL_RENDERER", "?")
        try:
            self.msaa = ctx.framebuffer(ctx.renderbuffer((self.w, self.h), samples=samples),
                                        ctx.depth_renderbuffer((self.w, self.h), samples=samples))
            self.samples = samples
        except Exception:                            # driver without MSAA renderbuffers
            self.msaa = ctx.framebuffer(ctx.renderbuffer((self.w, self.h)),
                                        ctx.depth_renderbuffer((self.w, self.h)))
            self.samples = 0
        self.out = ctx.framebuffer(ctx.renderbuffer((self.w, self.h)))
        self.holo = ctx.program(vertex_shader=HOLO_VS, fragment_shader=HOLO_FS)
        self.flat = ctx.program(vertex_shader=FLAT_VS, fragment_shader=FLAT_FS)
        self.pts = ctx.program(vertex_shader=PT_VS, fragment_shader=PT_FS)
        ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.prims: List[_GLPrim] = []
        self._build_objects()
        self._build_stage()
        rng = np.random.default_rng(7)
        self.n_pts = 420
        self.pt_state = np.stack([rng.uniform(-1.6, 1.6, self.n_pts), rng.uniform(0, 2.4, self.n_pts),
                                  rng.uniform(-1.6, 1.6, self.n_pts), rng.uniform(2.0, 6.0, self.n_pts)], 1).astype(np.float32)
        self.pt_vbo = ctx.buffer(reserve=self.n_pts * 16, dynamic=True)
        self.pt_vao = ctx.vertex_array(self.pts, [(self.pt_vbo, "4f", "in_p")])
        self._last_t = None

    # ------------------------------------------------------------ resources
    def _u(self, prog, name, value):
        if name in prog:
            prog[name].value = value

    def _m(self, prog, name, m):
        if name in prog:
            prog[name].write(np.ascontiguousarray(m.T, np.float32).tobytes())

    def load_rig(self, rig):
        ctx = self.ctx
        model = rig.model
        for k, prim in enumerate(model.primitives):
            g = _GLPrim()
            g.n_idx = len(prim.indices)
            g.vbo_pos = ctx.buffer(reserve=prim.positions.nbytes, dynamic=True)
            g.vbo_nor = ctx.buffer(reserve=prim.normals.nbytes, dynamic=True)
            g.vbo_uv = ctx.buffer(prim.uvs.astype(np.float32).tobytes())
            reg = np.where(rig.prim_region[k] < 0, -9.0, rig.prim_region[k]).astype(np.float32)
            g.vbo_reg = ctx.buffer(reg.tobytes())
            g.ibo = ctx.buffer(prim.indices.astype(np.uint32).tobytes())
            g.vao = ctx.vertex_array(self.holo, [(g.vbo_pos, "3f", "in_pos"), (g.vbo_nor, "3f", "in_nor"),
                                                 (g.vbo_uv, "2f", "in_uv"), (g.vbo_reg, "1f", "in_region")], g.ibo)
            mat = model.materials[prim.material] if 0 <= prim.material < len(model.materials) else None
            g.base = tuple(float(x) for x in (mat.base_color if mat else (1, 1, 1, 1)))
            g.alpha_mode = mat.alpha_mode if mat else "OPAQUE"
            g.cutoff = (mat.alpha_cutoff if mat.alpha_mode == "MASK" else 0.08) if (mat and mat.alpha_mode != "OPAQUE") else -1.0
            g.double_sided = bool(mat.double_sided) if mat else False
            g.tex = None
            if mat is not None and mat.texture is not None and model.images[mat.texture] is not None:
                img = model.images[mat.texture]
                h, w = img.shape[:2]
                t = ctx.texture((w, h), 4, np.ascontiguousarray(img).tobytes())
                t.build_mipmaps()
                t.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
                try:
                    t.anisotropy = 8.0
                except Exception:
                    pass
                g.tex = t
            self.prims.append(g)
        self.has_textures = any(g.tex is not None for g in self.prims)

    def update_meshes(self, skinned: List[Tuple[np.ndarray, np.ndarray]]):
        for g, (pos, nor) in zip(self.prims, skinned):
            g.vbo_pos.write(np.ascontiguousarray(pos, np.float32).tobytes())
            g.vbo_nor.write(np.ascontiguousarray(nor, np.float32).tobytes())

    def _build_objects(self):
        ctx = self.ctx
        self.obj = {}
        for kind, builder in MESH_BUILDERS.items():
            p, n, idx = builder()
            vp = ctx.buffer(p.tobytes())
            vn = ctx.buffer(n.tobytes())
            vu = ctx.buffer(np.zeros((len(p), 2), np.float32).tobytes())
            vr = ctx.buffer(np.full(len(p), -9.0, np.float32).tobytes())
            ib = ctx.buffer(idx.astype(np.uint32).tobytes())
            vao = ctx.vertex_array(self.holo, [(vp, "3f", "in_pos"), (vn, "3f", "in_nor"),
                                               (vu, "2f", "in_uv"), (vr, "1f", "in_region")], ib)
            self.obj[kind] = (vao, len(idx))

    def _flat_vao(self, verts: np.ndarray):
        buf = self.ctx.buffer(verts.astype(np.float32).tobytes())
        return self.ctx.vertex_array(self.flat, [(buf, "3f 4f", "in_pos", "in_col")]), len(verts)

    def _build_stage(self):
        # grid
        lines = []
        R = 2.6
        for x in np.arange(-R, R + 1e-6, 0.2):
            for (a, b) in (((x, 0, -R), (x, 0, R)), ((-R, 0, x), (R, 0, x))):
                for p in (a, b):
                    r = math.hypot(p[0], p[2]) / R
                    lines.append((*p, 0.35, 0.8, 1.0, 0.30 * max(0.0, 1 - r) ** 1.6))
        self.grid_vao, self.grid_n = self._flat_vao(np.array(lines, np.float32))

        def ring(r, seg=96, a0=0.0, a1=2 * math.pi, alpha=0.8, col=(0.4, 0.9, 1.0), dash=0):
            out = []
            for i in range(seg):
                if dash and (i // dash) % 2:
                    continue
                for j in (i, i + 1):
                    t = a0 + (a1 - a0) * j / seg
                    out.append((r * math.cos(t), 0.004, r * math.sin(t), *col, alpha))
            return out
        ra = ring(0.62, alpha=0.9) + ring(0.95, seg=120, dash=3, alpha=0.7) + ring(1.55, seg=120, dash=2, alpha=0.35)
        rb = ring(0.78, seg=100, a0=0.2, a1=2.6, alpha=0.9, col=(0.6, 1.0, 1.0)) + \
            ring(0.78, seg=100, a0=3.4, a1=5.8, alpha=0.9, col=(0.6, 1.0, 1.0)) + ring(1.25, seg=140, dash=1, alpha=0.5)
        self.ring_a, self.ring_a_n = self._flat_vao(np.array(ra, np.float32))
        self.ring_b, self.ring_b_n = self._flat_vao(np.array(rb, np.float32))
        # disc + light cone (triangles, additive)
        tri = []
        seg = 64
        for i in range(seg):
            a, b = 2 * math.pi * i / seg, 2 * math.pi * (i + 1) / seg
            tri += [(0, 0.002, 0, 0.2, 0.7, 1.0, 0.22), (0.7 * math.cos(a), 0.002, 0.7 * math.sin(a), 0.2, 0.7, 1.0, 0.0),
                    (0.7 * math.cos(b), 0.002, 0.7 * math.sin(b), 0.2, 0.7, 1.0, 0.0)]
            r0, r1, h = 0.72, 0.52, 2.35
            p0a, p0b = (r0 * math.cos(a), 0.0, r0 * math.sin(a)), (r0 * math.cos(b), 0.0, r0 * math.sin(b))
            p1a, p1b = (r1 * math.cos(a), h, r1 * math.sin(a)), (r1 * math.cos(b), h, r1 * math.sin(b))
            c0, c1 = (0.2, 0.75, 1.0, 0.10), (0.2, 0.75, 1.0, 0.0)
            tri += [(*p0a, *c0), (*p0b, *c0), (*p1a, *c1), (*p0b, *c0), (*p1b, *c1), (*p1a, *c1)]
        self.cone_vao, self.cone_n = self._flat_vao(np.array(tri, np.float32))

    # ------------------------------------------------------------ drawing
    def _hologram_pass(self, rs: RenderState, model, mode, mirrored=False, reflect=0.0):
        ctx, p = self.ctx, self.holo
        self._m(p, "u_model", model)
        self._u(p, "u_mode", mode)
        self._u(p, "u_reflect", reflect)
        self._u(p, "u_fade", 0.0 if mirrored else 1.0)
        self._u(p, "u_wire", 1.0 if mode == 1 else 0.0)
        for g in self.prims:
            if g.double_sided:
                ctx.disable(moderngl.CULL_FACE)
            else:
                ctx.enable(moderngl.CULL_FACE)
            self._u(p, "u_base", g.base)
            self._u(p, "u_cutoff", g.cutoff)
            if g.tex is not None:
                g.tex.use(0)
                self._u(p, "u_tex", 0)
                self._u(p, "u_has_tex", 1)
            else:
                self._u(p, "u_has_tex", 0)
            self._u(p, "u_texmix", 0.88)
            g.vao.render()
        ctx.disable(moderngl.CULL_FACE)

    def render(self, rs: RenderState) -> np.ndarray:
        """Draw one frame. Returns a BGR uint8 image (h, w, 3): additive light on black."""
        ctx = self.ctx
        t = rs.time
        dt = 0.016 if self._last_t is None else max(min(t - self._last_t, 0.1), 0.0)
        self._last_t = t
        cam = rs.camera
        fbo = self.msaa
        fbo.use()
        ctx.viewport = (0, 0, self.w, self.h)
        fbo.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        ctx.wireframe = False
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.depth_func = "<="
        h, f = self.holo, self.flat
        for prog in (h, f, self.pts):
            self._m(prog, "u_vp", cam.vp)
        self._u(h, "u_cam", tuple(float(x) for x in cam.eye))
        self._u(h, "u_time", float(t))
        self._u(h, "u_glitch", float(rs.glitch))
        dim = 1.0 - 0.55 * rs.standby
        self._u(h, "u_alpha", float(rs.alpha * dim))
        self._u(h, "u_tint", rs.tint)
        self._u(h, "u_tint2", rs.tint2)
        self._u(h, "u_hl_a", float(rs.hover_region))
        self._u(h, "u_hl_b", float(rs.selected_region))
        self._u(h, "u_hl_a_amt", float(rs.hover_amt))
        self._u(h, "u_hl_b_amt", float(rs.selected_amt))
        self._u(h, "u_hl_col", (0.25, 0.55, 1.0))

        robot = rs.robot_model
        mirror = (robot @ np.diag([1, -1, 1, 1]).astype(np.float32)).astype(np.float32)
        # 1) depth pre-pass: only the front-most surface survives -> solid-looking hologram
        fbo.color_mask = (False, False, False, False)
        fbo.depth_mask = True
        self._hologram_pass(rs, robot, 2)
        ctx.front_face = "cw"
        self._hologram_pass(rs, mirror, 2, mirrored=True)
        ctx.front_face = "ccw"
        fbo.color_mask = (True, True, True, True)
        fbo.depth_mask = False
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE)
        # 2) stage: grid, discs, rings, light cone
        self._m(f, "u_model", rs.floor_model)
        self._u(f, "u_tint", (1.0, 1.0, 1.0))
        self._u(f, "u_alpha", float(1.0 * dim))
        self.grid_vao.render(moderngl.LINES, vertices=self.grid_n)
        self.cone_vao.render(moderngl.TRIANGLES, vertices=self.cone_n)
        self._m(f, "u_model", rs.floor_model @ rot_y(t * 0.35))
        self.ring_a.render(moderngl.LINES, vertices=self.ring_a_n)
        self._m(f, "u_model", rs.floor_model @ rot_y(-t * 0.6))
        self.ring_b.render(moderngl.LINES, vertices=self.ring_b_n)
        # 3) the hologram: solid + wire, then the reflection
        self._hologram_pass(rs, robot, 0)
        if self.wire:
            ctx.wireframe = True
            self._u(h, "u_alpha", float(rs.alpha * dim * 0.025))
            self._hologram_pass(rs, robot, 1)
            ctx.wireframe = False
            self._u(h, "u_alpha", float(rs.alpha * dim))
        ctx.front_face = "cw"
        self._hologram_pass(rs, mirror, 0, mirrored=True, reflect=0.28)
        ctx.front_face = "ccw"
        # 4) interactive objects
        for o in rs.objects:
            vao, n = self.obj[o.kind]
            self._m(h, "u_model", o.model_matrix(t))
            col = np.array(o.color, np.float32)
            glow = 0.55 * o.hover + (0.5 if o.selected else 0.0) + (0.4 if o.grabbed else 0.0)
            self._u(h, "u_tint", tuple(float(x) for x in np.clip(col * (0.7 + glow), 0, 1.6)))
            self._u(h, "u_tint2", tuple(float(x) for x in np.clip(col * 0.6 + 0.5 + 0.3 * glow, 0, 1.6)))
            self._u(h, "u_has_tex", 0)
            self._u(h, "u_base", (1.0, 1.0, 1.0, 1.0))
            self._u(h, "u_cutoff", -1.0)
            self._u(h, "u_texmix", 0.0)
            self._u(h, "u_hl_a_amt", 0.0)
            self._u(h, "u_hl_b_amt", 0.0)
            self._u(h, "u_fade", 0.0)
            self._u(h, "u_reflect", 0.0)
            self._u(h, "u_mode", 0)
            self._u(h, "u_wire", 0.0)
            self._u(h, "u_alpha", float(0.42 + 0.25 * glow))
            ctx.enable(moderngl.CULL_FACE)
            vao.render(moderngl.TRIANGLES, vertices=n)
            ctx.wireframe = True
            self._u(h, "u_mode", 1)
            self._u(h, "u_wire", 1.0)
            self._u(h, "u_alpha", float(0.85 + 0.3 * glow))
            ctx.disable(moderngl.CULL_FACE)
            vao.render(moderngl.TRIANGLES, vertices=n)
            ctx.wireframe = False
        # 5) floating light particles
        self._advance_particles(dt, t)
        self._u(self.pts, "u_scale", 4.5 * self.h / 720.0)
        self._u(self.pts, "u_col", (0.5, 0.9, 1.0))
        self._u(self.pts, "u_alpha", float(0.55 * dim))
        self.pt_vao.render(moderngl.POINTS, vertices=self.n_pts)
        ctx.disable(moderngl.BLEND)
        fbo.depth_mask = True
        # 6) resolve MSAA and read back
        ctx.copy_framebuffer(self.out, self.msaa)
        data = self.out.read(components=3, alignment=1)
        img = np.frombuffer(data, np.uint8).reshape(self.h, self.w, 3)[::-1]
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def _advance_particles(self, dt, t):
        s = self.pt_state
        s[:, 1] += dt * (0.06 + 0.04 * (s[:, 3] % 1.0))
        s[:, 0] += dt * 0.03 * np.sin(t * 0.7 + s[:, 3] * 3.0)
        wrap = s[:, 1] > 2.5
        s[wrap, 1] = 0.0
        self.pt_vbo.write(s.astype(np.float32).tobytes())

    def close(self):
        try:
            self.ctx.release()
        except Exception:
            pass