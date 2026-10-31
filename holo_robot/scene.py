"""Scene description: virtual camera, interactive holographic objects, procedural meshes, picking."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .mathutil import look_at, normalize, perspective, ray_sphere, rot_x, rot_y, scale_m, translate


# ---------------------------------------------------------------- camera
class SceneCamera:
    """Perspective camera looking at the hologram. Provides project / unproject for picking."""

    def __init__(self, width: int, height: int, fov=36.0):
        self.w, self.h, self.fov = width, height, fov
        self.eye = np.array([0.0, 1.30, 4.1], np.float32)
        self.target = np.array([0.0, 0.92, 0.0], np.float32)
        self.update()

    def update(self):
        self.view = look_at(self.eye, self.target)
        self.proj = perspective(self.fov, self.w / self.h, 0.1, 30.0)
        self.vp = (self.proj @ self.view).astype(np.float32)
        self.inv_vp = np.linalg.inv(self.vp).astype(np.float32)

    def zoom(self, factor: float):
        """Zoom manually around the current look-at target, clamped to a useful range."""
        direction = self.eye - self.target
        distance = float(np.linalg.norm(direction))
        new_distance = float(np.clip(distance * factor, 1.8, 8.0))
        self.eye = self.target + normalize(direction) * new_distance
        self.update()

    def focus(self, point):
        """Move the look-at target to a manually selected world-space point."""
        offset = np.asarray(point, np.float32) - self.target
        self.target = np.asarray(point, np.float32).copy()
        self.eye += offset
        self.update()

    def reset(self):
        self.eye = np.array([0.0, 1.30, 4.1], np.float32)
        self.target = np.array([0.0, 0.92, 0.0], np.float32)
        self.update()

    def ray(self, px: float, py: float) -> Tuple[np.ndarray, np.ndarray]:
        """Pixel -> world ray (origin, unit direction)."""
        x = 2.0 * px / self.w - 1.0
        y = 1.0 - 2.0 * py / self.h
        far = self.inv_vp @ np.array([x, y, 1.0, 1.0], np.float32)
        near = self.inv_vp @ np.array([x, y, -1.0, 1.0], np.float32)
        far, near = far[:3] / far[3], near[:3] / near[3]
        return self.eye.copy(), normalize(far - near)

    def project(self, p) -> Optional[Tuple[float, float, float]]:
        """World point -> (px, py, depth) or None if behind the camera."""
        v = self.vp @ np.array([p[0], p[1], p[2], 1.0], np.float32)
        if v[3] <= 1e-6:
            return None
        return (float((v[0] / v[3] * 0.5 + 0.5) * self.w), float((0.5 - v[1] / v[3] * 0.5) * self.h),
                float(v[2] / v[3]))

    def forward(self):
        return normalize(self.target - self.eye)


# ---------------------------------------------------------------- procedural meshes
def _flat(pos, idx):
    """Expand indexed mesh to flat-shaded (per-face normals)."""
    tri = idx.reshape(-1, 3)
    p = pos[tri.reshape(-1)]
    fn = np.cross(p[1::3] - p[0::3], p[2::3] - p[0::3])
    fn = fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-9)
    nor = np.repeat(fn, 3, axis=0)
    return p.astype(np.float32), nor.astype(np.float32), np.arange(len(p), dtype=np.uint32)


def mesh_cube():
    v = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                  [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], np.float32) * 0.5
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [2, 3, 7], [2, 7, 6], [1, 2, 6], [1, 6, 5], [0, 4, 7], [0, 7, 3]], np.uint32)
    return _flat(v, f.reshape(-1))


def mesh_octa():
    v = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], np.float32) * 0.6
    f = np.array([[0, 2, 4], [4, 2, 1], [1, 2, 5], [5, 2, 0], [4, 3, 0], [1, 3, 4], [5, 3, 1], [0, 3, 5]], np.uint32)
    return _flat(v, f.reshape(-1))


def mesh_icosphere(subdiv=1):
    t = (1 + math.sqrt(5)) / 2
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
         (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2),
         (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5),
         (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    v = [tuple(np.array(p, np.float32) / np.linalg.norm(p)) for p in v]
    for _ in range(subdiv):
        cache, nf = {}, []

        def mid(a, b):
            k = (min(a, b), max(a, b))
            if k not in cache:
                m = (np.array(v[a]) + np.array(v[b])) / 2
                v.append(tuple(m / np.linalg.norm(m)))
                cache[k] = len(v) - 1
            return cache[k]
        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        f = nf
    return _flat(np.array(v, np.float32) * 0.5, np.array(f, np.uint32).reshape(-1))


def mesh_torus(R=0.42, r=0.14, seg=28, side=12):
    pts, idx = [], []
    for i in range(seg):
        for j in range(side):
            a, b = 2 * math.pi * i / seg, 2 * math.pi * j / side
            pts.append(((R + r * math.cos(b)) * math.cos(a), r * math.sin(b), (R + r * math.cos(b)) * math.sin(a)))
    for i in range(seg):
        for j in range(side):
            a, b = i * side + j, ((i + 1) % seg) * side + j
            a2, b2 = i * side + (j + 1) % side, ((i + 1) % seg) * side + (j + 1) % side
            idx += [a, b, a2, b, b2, a2]
    return _flat(np.array(pts, np.float32), np.array(idx, np.uint32))


MESH_BUILDERS = {"cube": mesh_cube, "octa": mesh_octa, "sphere": mesh_icosphere, "torus": mesh_torus}


# ---------------------------------------------------------------- objects
@dataclass
class SceneObject:
    name: str
    kind: str
    position: np.ndarray
    color: Tuple[float, float, float]
    size: float = 0.32
    description: str = ""
    spin: float = 0.6
    # runtime state
    angle: float = 0.0
    hover: float = 0.0           # 0..1 smoothed
    selected: bool = False
    grabbed: bool = False
    base_position: np.ndarray = field(default=None)

    def __post_init__(self):
        self.position = np.asarray(self.position, np.float32)
        self.base_position = self.position.copy()

    @property
    def radius(self):
        return self.size * 0.75

    def model_matrix(self, t: float):
        bob = 0.0 if self.grabbed else 0.03 * math.sin(t * 1.4 + self.base_position[0] * 3)
        m = translate(self.position[0], self.position[1] + bob, self.position[2])
        tilt = rot_x(1.15) if self.kind == "torus" else np.eye(4, dtype=np.float32)   # ring faces the viewer
        return (m @ rot_y(self.angle) @ tilt @ scale_m(self.size * (1.0 + 0.12 * self.hover + (0.1 if self.selected else 0)))).astype(np.float32)


def default_objects() -> List[SceneObject]:
    return [
        SceneObject("DATA CUBE", "cube", (-1.0, 1.45, 0.25), (0.35, 0.85, 1.0), 0.30,
                    "Interactive storage cube. Pinch to move it."),
        SceneObject("ENERGY CORE", "sphere", (1.0, 1.50, 0.25), (1.0, 0.55, 0.25), 0.36,
                    "Plasma sphere. Point at it and the robot reaches for it."),
        SceneObject("RING GATE", "torus", (-0.95, 0.60, 0.45), (0.55, 1.0, 0.65), 0.55,
                    "Portal ring. Drag it anywhere on the stage.", spin=1.1),
        SceneObject("SIGNAL CRYSTAL", "octa", (0.95, 0.55, 0.45), (0.95, 0.45, 1.0), 0.40,
                    "Resonant crystal. Select it to see its state."),
    ]


class Scene:
    """Holds the hologram transform and the interactive objects."""

    def __init__(self, camera: SceneCamera):
        self.camera = camera
        self.objects = default_objects()
        self.robot_pos = np.zeros(3, np.float32)
        self.robot_yaw = 0.0
        self.robot_scale = 1.0
        self.regions: List[Tuple[str, np.ndarray, float]] = []   # world-space colliders

    def robot_matrix(self, fit: np.ndarray) -> np.ndarray:
        """model/rig space -> world."""
        return (translate(*self.robot_pos) @ rot_y(self.robot_yaw) @ scale_m(self.robot_scale) @ fit).astype(np.float32)

    def update(self, dt: float, t: float):
        for o in self.objects:
            o.angle += o.spin * dt * (0.25 if o.grabbed else 1.0)

    def pick(self, px: float, py: float):
        """What is under pixel (px,py)? -> ('object', obj, dist) | ('region', idx, dist) | None"""
        o, d = self.camera.ray(px, py)
        best = None
        for obj in self.objects:
            h = ray_sphere(o, d, obj.position, obj.radius * 1.25)
            if h is not None and (best is None or h < best[2]):
                best = ("object", obj, h)
        for i, (name, c, r) in enumerate(self.regions):
            h = ray_sphere(o, d, c, r)
            if h is not None and (best is None or h < best[2] - 0.02):
                best = ("region", i, h)
        return best


# ---------------------------------------------------------------- per-frame render description
@dataclass
class RenderState:
    """Everything a renderer needs to draw one frame."""
    time: float
    camera: SceneCamera
    robot_model: np.ndarray                       # rig space -> world
    floor_model: np.ndarray
    objects: List[SceneObject]
    hover_region: int = -1
    selected_region: int = -1
    hover_amt: float = 0.0
    selected_amt: float = 0.0
    glitch: float = 0.15
    tint: Tuple[float, float, float] = (1.00, 1.00, 1.00)
    tint2: Tuple[float, float, float] = (0.35, 0.78, 1.00)
    alpha: float = 0.9
    standby: float = 0.0                          # 0..1 dims the hologram