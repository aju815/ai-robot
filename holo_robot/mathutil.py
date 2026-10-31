"""Small linear-algebra toolbox (column-vector convention, right-handed, Y up)."""
from __future__ import annotations

import math

import numpy as np

EPS = 1e-9


def normalize(v):
    v = np.asarray(v, np.float32)
    n = np.linalg.norm(v)
    return v / n if n > EPS else v * 0.0


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz], np.float32)


def quat_conj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]], np.float32)


def quat_from_axis_angle(axis, angle):
    axis = normalize(axis)
    s = math.sin(angle * 0.5)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(angle * 0.5)], np.float32)


def slerp(a, b, t):
    """Spherical interpolation. Works for single quaternions (4,) and batches (N,4)."""
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    dot = np.sum(a * b, axis=-1, keepdims=True)
    b = np.where(dot < 0, -b, b)
    dot = np.abs(dot)
    lin = dot > 0.9995
    theta = np.arccos(np.clip(dot, -1, 1))
    sin_t = np.sin(theta)
    sin_t = np.where(sin_t < 1e-6, 1.0, sin_t)
    wa = np.where(lin, 1.0 - t, np.sin((1.0 - t) * theta) / sin_t)
    wb = np.where(lin, t, np.sin(t * theta) / sin_t)
    out = wa * a + wb * b
    return out / np.maximum(np.linalg.norm(out, axis=-1, keepdims=True), EPS)


def quat_to_mat3(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], np.float32)


def mat3_to_quat(m):
    m = np.asarray(m, np.float64)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    q = np.array(q, np.float32)
    return q / (np.linalg.norm(q) + EPS)


def compose_trs(t, q, s):
    m = np.eye(4, dtype=np.float32)
    m[:3, :3] = quat_to_mat3(q) * np.asarray(s, np.float32)[None, :]
    m[:3, 3] = t
    return m


def rot_axis_angle_mat3(axis, angle):
    return quat_to_mat3(quat_from_axis_angle(axis, angle))


def rotation_between(a, b):
    """Shortest-arc rotation (3x3) taking direction a onto direction b."""
    a, b = normalize(a), normalize(b)
    c = float(np.dot(a, b))
    if c > 1 - 1e-7:
        return np.eye(3, dtype=np.float32)
    if c < -1 + 1e-7:
        ax = np.cross(a, [1, 0, 0])
        if np.linalg.norm(ax) < 1e-3:
            ax = np.cross(a, [0, 1, 0])
        return rot_axis_angle_mat3(ax, math.pi)
    axis = np.cross(a, b)
    return rot_axis_angle_mat3(axis, math.acos(np.clip(c, -1, 1)))


def limit_rotation(rot3, max_angle, weight=1.0):
    """Scale a 3x3 rotation to at most max_angle radians, then by weight."""
    q = mat3_to_quat(rot3)
    if q[3] < 0:
        q = -q
    angle = 2 * math.acos(np.clip(q[3], -1, 1))
    s = math.sqrt(max(1 - q[3] * q[3], 0.0))
    if s < 1e-6 or angle < 1e-6:
        return np.eye(3, dtype=np.float32)
    axis = q[:3] / s
    return rot_axis_angle_mat3(axis, min(angle, max_angle) * weight)


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4, dtype=np.float32)
    m[0, 0], m[0, 2], m[2, 0], m[2, 2] = c, s, -s, c
    return m


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4, dtype=np.float32)
    m[1, 1], m[1, 2], m[2, 1], m[2, 2] = c, -s, s, c
    return m


def translate(x, y, z):
    m = np.eye(4, dtype=np.float32)
    m[:3, 3] = (x, y, z)
    return m


def scale_m(s):
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = m[1, 1] = m[2, 2] = s
    return m


def perspective(fovy_deg, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fovy_deg) / 2)
    m = np.zeros((4, 4), np.float32)
    m[0, 0], m[1, 1] = f / aspect, f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2 * far * near / (near - far)
    m[3, 2] = -1
    return m


def look_at(eye, target, up=(0, 1, 0)):
    eye, target = np.asarray(eye, np.float32), np.asarray(target, np.float32)
    f = normalize(target - eye)
    s = normalize(np.cross(f, up))
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def ray_sphere(origin, direction, center, radius):
    """Distance along the ray to the first hit with the sphere, or None."""
    oc = origin - center
    b = float(np.dot(oc, direction))
    c = float(np.dot(oc, oc) - radius * radius)
    disc = b * b - c
    if disc < 0:
        return None
    t = -b - math.sqrt(disc)
    if t < 0:
        t = -b + math.sqrt(disc)
    return t if t >= 0 else None


class OneEuroFilter:
    """Adaptive low-pass filter (Casiez et al.): smooth when slow, responsive when fast.
    Removes hand-tracking jitter without adding lag to quick motions."""

    def __init__(self, min_cutoff=1.2, beta=0.03, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = None
        self.dx = None
        self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def reset(self):
        self.x = self.dx = self.t = None

    def __call__(self, x, t):
        x = np.asarray(x, np.float32)
        if self.x is None:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x
        dt = max(t - self.t, 1e-4)
        self.t = t
        dx = (x - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self.dx))
        a = self._alpha(cutoff, dt)
        self.x = a * x + (1 - a) * self.x
        return self.x.copy()