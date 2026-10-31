"""Skeleton, animation playback, retargeting, skinning and procedural body control."""
from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from .gltf_loader import GltfModel, Clip
from .mathutil import (compose_trs, limit_rotation, mat3_to_quat, normalize, quat_from_axis_angle,
                       quat_mul, quat_to_mat3, rotation_between, slerp)

FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def norm_name(name: str) -> str:
    """'mixamorig:RightHandIndex1' / 'mixamorigRightHandIndex1' -> 'righthandindex1'."""
    n = name.lower()
    n = re.sub(r"^.*?mixamorig:?", "", n)
    return n.split(":")[-1].replace("_", "").replace(".", "").replace(" ", "")


class Rig:
    """Node hierarchy + current local pose + world matrices + CPU skinning."""

    def __init__(self, model: GltfModel):
        self.model = model
        nodes = model.nodes
        n = len(nodes)
        self.n = n
        self.parent = np.array([nd.parent for nd in nodes], np.int64)
        self.names = [nd.name for nd in nodes]
        self.by_norm: Dict[str, int] = {}
        for i, nm in enumerate(self.names):
            self.by_norm.setdefault(norm_name(nm), i)
        self.bind_t = np.array([nd.t for nd in nodes], np.float32)
        self.bind_r = np.array([nd.r for nd in nodes], np.float32)
        self.bind_s = np.array([nd.s for nd in nodes], np.float32)
        self.t, self.r, self.s = self.bind_t.copy(), self.bind_r.copy(), self.bind_s.copy()
        self.world = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
        # topological order (parents first)
        order, seen = [], set()

        def visit(i):
            if i in seen:
                return
            seen.add(i)
            order.append(i)
            for c in nodes[i].children:
                visit(c)
        for r in model.roots:
            visit(r)
        for i in range(n):
            visit(i)
        self.order = order
        self.descendants: List[List[int]] = [[] for _ in range(n)]
        for i in order:
            p = i
            while p >= 0:
                self.descendants[p].append(i)
                p = self.parent[p]
        for d in self.descendants:
            d.sort(key=order.index)
        self.update_world()
        self.bind_world = self.world.copy()
        self.roles = self._find_roles()
        self.facing = self._detect_facing()
        self.region_of_node = self._assign_regions()
        self._prepare_vertex_regions()
        self.fit_scale, self.fit_offset = self._compute_fit()
        self._finger_axes = self._compute_finger_axes()
        self._head_forward = self._local_forward("head")

    # ------------------------------------------------------------ pose / world
    def reset_pose(self):
        self.t[:], self.r[:], self.s[:] = self.bind_t, self.bind_r, self.bind_s

    def update_world(self, root: Optional[int] = None):
        idxs = self.order if root is None else self.descendants[root]
        for i in idxs:
            local = compose_trs(self.t[i], self.r[i], self.s[i])
            p = self.parent[i]
            self.world[i] = local if p < 0 else self.world[p] @ local

    def node_pos(self, i):
        return self.world[i][:3, 3]

    def node_rot3(self, i):
        m = self.world[i][:3, :3]
        return m / np.maximum(np.linalg.norm(m, axis=0, keepdims=True), 1e-9)

    # ------------------------------------------------------------ roles / regions
    def _find_roles(self) -> Dict[str, int]:
        g = self.by_norm.get
        roles: Dict[str, int] = {}
        simple = {"hips": "hips", "spine": "spine", "spine2": "chest", "neck": "neck", "head": "head",
                  "headtopend": "headtop"}
        for key, role in simple.items():
            if key in self.by_norm:
                roles[role] = g(key)
        for side in ("left", "right"):
            for part, nm in (("shoulder", "shoulder"), ("arm", "arm"), ("forearm", "forearm"),
                             ("hand", "hand"), ("upleg", "upleg"), ("leg", "leg"), ("foot", "foot")):
                k = side + nm
                if k in self.by_norm:
                    roles[f"{side}_{part}"] = g(k)
            for f in FINGERS:
                chain = [g(f"{side}hand{f}{j}") for j in (1, 2, 3)]
                if all(c is not None for c in chain):
                    roles[f"{side}_{f}"] = chain  # type: ignore[assignment]
        return roles

    @property
    def has_humanoid(self) -> bool:
        need = ("hips", "head", "left_arm", "left_forearm", "left_hand",
                "right_arm", "right_forearm", "right_hand")
        return all(k in self.roles for k in need)

    REGION_NAMES = ["HEAD UNIT", "TORSO CORE", "LEFT ARM", "RIGHT ARM", "LEFT LEG", "RIGHT LEG"]

    def _assign_regions(self) -> np.ndarray:
        """Region id per node (0 head,1 torso,2 L arm,3 R arm,4 L leg,5 R leg), -1 if unknown."""
        rg = np.full(self.n, -1, np.int64)
        r = self.roles
        marks = []
        if "hips" in r:
            marks.append((r["hips"], 1))
        if "spine" in r:
            marks.append((r["spine"], 1))
        for key, rid in (("neck", 0), ("head", 0), ("left_shoulder", 2), ("left_arm", 2),
                         ("right_shoulder", 3), ("right_arm", 3), ("left_upleg", 4), ("right_upleg", 5)):
            if key in r:
                marks.append((r[key], rid))
        for node, rid in marks:                           # deeper marks are applied later
            for d in self.descendants[node]:
                rg[d] = rid
        return rg

    def _prepare_vertex_regions(self):
        self.prim_region = []
        for prim in self.model.primitives:
            node = self.model.nodes[prim.node]
            if prim.joints is None or node.skin is None:
                rid = self.region_of_node[prim.node]
                self.prim_region.append(np.full(len(prim.positions), float(rid), np.float32))
                continue
            joints = np.array(self.model.skins[node.skin].joints)
            jr = self.region_of_node[joints]
            dom = prim.joints[np.arange(len(prim.joints)), np.argmax(prim.weights, axis=1)]
            self.prim_region.append(jr[dom].astype(np.float32))

    def region_centers(self) -> List[Tuple[str, np.ndarray, float]]:
        """(name, centre in model space, radius in model units) used for hover picking."""
        out = []
        r = self.roles
        h = self.height_model
        def pos(k):
            return self.node_pos(r[k]) if k in r else None
        head = pos("head")
        if head is not None:
            out.append((self.REGION_NAMES[0], head + np.array([0, 0.05 * h, 0], np.float32), 0.075 * h))
        if "hips" in r and "chest" in r:
            out.append((self.REGION_NAMES[1], (pos("hips") + pos("chest")) / 2 + np.array([0, 0.02 * h, 0], np.float32), 0.14 * h))
        for side, rid in (("left", 2), ("right", 3)):
            if f"{side}_arm" in r and f"{side}_hand" in r:
                out.append((self.REGION_NAMES[rid], (pos(f"{side}_arm") + pos(f"{side}_hand")) / 2, 0.11 * h))
        for side, rid in (("left", 4), ("right", 5)):
            if f"{side}_upleg" in r and f"{side}_foot" in r:
                out.append((self.REGION_NAMES[rid], (pos(f"{side}_upleg") + pos(f"{side}_foot")) / 2, 0.14 * h))
        return out

    # ------------------------------------------------------------ fit to 1.8 m
    def _compute_fit(self, target=1.8):
        pts = []
        for prim, _ in zip(self.model.primitives, range(len(self.model.primitives))):
            pos, _ = self.skin_primitive(prim)
            pts.append(pos)
        allp = np.concatenate(pts)
        lo, hi = allp.min(0), allp.max(0)
        self.height_model = float(hi[1] - lo[1])
        scale = target / max(self.height_model, 1e-6)
        offset = np.array([-(lo[0] + hi[0]) / 2, -lo[1], -(lo[2] + hi[2]) / 2], np.float32)
        return float(scale), offset

    def _detect_facing(self):
        """Direction the character faces at bind pose (from its left/right arm positions)."""
        r = self.roles
        if "left_arm" in r and "right_arm" in r:
            lr = self.bind_world[r["left_arm"]][:3, 3] - self.bind_world[r["right_arm"]][:3, 3]
            f = np.cross(lr, np.array([0, 1, 0], np.float32))
            f[1] = 0
            if np.linalg.norm(f) > 1e-6:
                return normalize(f)
        return np.array([0, 0, 1], np.float32)

    def fit_matrix(self):
        """rig space -> metre-sized, feet-on-floor, facing +Z (towards the viewer)."""
        yaw = math.atan2(self.facing[0], self.facing[2])       # rotate so facing becomes +Z
        c, s = math.cos(-yaw), math.sin(-yaw)
        rot = np.eye(4, dtype=np.float32)
        rot[0, 0], rot[0, 2], rot[2, 0], rot[2, 2] = c, s, -s, c
        off = np.eye(4, dtype=np.float32)
        off[:3, 3] = self.fit_offset
        sc = np.eye(4, dtype=np.float32)
        sc[0, 0] = sc[1, 1] = sc[2, 2] = self.fit_scale
        return sc @ rot @ off

    # ------------------------------------------------------------ skinning
    def skin_primitive(self, prim) -> Tuple[np.ndarray, np.ndarray]:
        node = self.model.nodes[prim.node]
        if prim.joints is None or node.skin is None:
            m = self.world[prim.node]
            pos = prim.positions @ m[:3, :3].T + m[:3, 3]
            nor = prim.normals @ np.linalg.inv(m[:3, :3])
            return pos.astype(np.float32), _unit(nor)
        skin = self.model.skins[node.skin]
        pal = self.world[np.array(skin.joints)] @ skin.inverse_bind          # (J,4,4)
        pos = np.zeros_like(prim.positions)
        nor = np.zeros_like(prim.normals)
        for k in range(4):
            w = prim.weights[:, k:k + 1]
            m = pal[prim.joints[:, k]]                                        # (N,4,4)
            pos += w * (np.einsum("nij,nj->ni", m[:, :3, :3], prim.positions) + m[:, :3, 3])
            nor += w * np.einsum("nij,nj->ni", m[:, :3, :3], prim.normals)
        return pos.astype(np.float32), _unit(nor)

    # ------------------------------------------------------------ procedural layers
    def _local_forward(self, role):
        if role not in self.roles:
            return None
        i = self.roles[role]
        return (self.bind_world[i][:3, :3] / np.maximum(
            np.linalg.norm(self.bind_world[i][:3, :3], axis=0, keepdims=True), 1e-9)).T @ self.facing

    def _apply_world_rotation(self, i, delta3):
        """Rotate node i by a world-space rotation delta (3x3), keeping the pose otherwise."""
        p = self.parent[i]
        rp = np.eye(3, dtype=np.float32) if p < 0 else self.node_rot3(p)
        rl = quat_to_mat3(self.r[i])
        new_local = rp.T @ delta3 @ rp @ rl
        self.r[i] = mat3_to_quat(new_local)
        self.update_world(i)

    def look_at(self, target_model, weight, max_angle=math.radians(65)):
        """Turn neck + head so the face looks at a point (model space)."""
        if "head" not in self.roles or self._head_forward is None or weight <= 1e-3:
            return
        head = self.roles["head"]
        for node, share in ((self.roles.get("neck"), 0.4), (head, 0.6)):
            if node is None:
                continue
            fwd = self.node_rot3(head) @ self._head_forward
            want = target_model - self.node_pos(head)
            d = rotation_between(fwd, want)
            self._apply_world_rotation(node, limit_rotation(d, max_angle * share, weight * share))

    def aim_arm(self, side, target_model, weight):
        """Straighten the arm toward a point (upper arm then forearm)."""
        r = self.roles
        keys = (f"{side}_arm", f"{side}_forearm", f"{side}_hand")
        if weight <= 1e-3 or not all(k in r for k in keys):
            return
        for a, b in ((keys[0], keys[1]), (keys[1], keys[2])):
            ia, ib = r[a], r[b]
            cur = self.node_pos(ib) - self.node_pos(ia)
            want = target_model - self.node_pos(ia)
            d = rotation_between(cur, want)
            self._apply_world_rotation(ia, limit_rotation(d, math.radians(140), weight))

    def hand_reach_point(self, side):
        return self.node_pos(self.roles[f"{side}_hand"]) if f"{side}_hand" in self.roles else None

    def _compute_finger_axes(self):
        """Local bend axis for every finger joint (fingers curl toward the palm).
        Derived from the bind pose, where Mixamo-style palms face down."""
        axes = {}
        palm = np.array([0, -1, 0], np.float32)
        for side in ("left", "right"):
            for f in FINGERS:
                chain = self.roles.get(f"{side}_{f}")
                if not chain:
                    continue
                for j, node in enumerate(chain):
                    child = chain[j + 1] if j + 1 < len(chain) else None
                    if child is None:
                        ch = [c for c in self.model.nodes[node].children]
                        child = ch[0] if ch else None
                    if child is None:
                        continue
                    d = normalize(self.bind_world[child][:3, 3] - self.bind_world[node][:3, 3])
                    axis_w = normalize(np.cross(d, palm))
                    if f == "thumb":
                        axis_w = normalize(np.cross(d, normalize(palm + np.array([1 if side == "left" else -1, 0, 1], np.float32) * 0.0)))
                    rw = self.bind_world[node][:3, :3] / np.maximum(
                        np.linalg.norm(self.bind_world[node][:3, :3], axis=0, keepdims=True), 1e-9)
                    axes[node] = (rw.T @ axis_w).astype(np.float32)
        return axes

    def curl_fingers(self, side, curls, weight=1.0):
        """curls: 5 values (thumb..pinky) in 0..1, 0 = straight, 1 = fully curled."""
        for f, c in zip(FINGERS, curls):
            chain = self.roles.get(f"{side}_{f}")
            if not chain:
                continue
            c = float(np.clip(c, 0, 1)) * weight
            amounts = (0.5, 1.05, 0.8) if f != "thumb" else (0.15, 0.55, 0.65)
            for node, a in zip(chain, amounts):
                ax = self._finger_axes.get(node)
                if ax is None:
                    continue
                self.r[node] = quat_mul(self.r[node], quat_from_axis_angle(ax, a * c * 1.35))
        hand = self.roles.get(f"{side}_hand")
        if hand is not None:
            self.update_world(hand)


def _unit(v):
    return (v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)).astype(np.float32)


# ====================================================================== animation
class RetargetedClip:
    """A clip whose channels are mapped by bone name onto a target rig.
    Rotations transfer directly (Mixamo-compatible skeletons). Root (hips) translation is
    converted through world space, so rigs exported with different axis conventions and
    different sizes still move correctly."""

    def __init__(self, clip: Clip, src_names: List[str], rig: Rig, src_hips: Optional[tuple]):
        self.name = clip.name
        self.duration = max(clip.duration, 1e-3)
        self.tracks = []        # (node, path, times, values, interp)
        hips = rig.roles.get("hips")
        self.hips = hips
        tp = rig.parent[hips] if hips is not None else -1
        self.P = rig.bind_world[tp][:3, :3].copy() if tp >= 0 else np.eye(3, dtype=np.float32)
        self.Pinv = np.linalg.inv(self.P).astype(np.float32)
        self.bind_wp = self.P @ rig.bind_t[hips] if hips is not None else None
        for ch in clip.channels:
            tgt = rig.by_norm.get(norm_name(src_names[ch.node]))
            if tgt is None:
                continue
            vals = ch.values
            if ch.path == "translation":
                if tgt != hips:
                    continue                      # only root motion is retargeted
                if src_hips is not None:
                    src_h, src_p = src_hips       # donor hips world height, donor parent 3x3
                    ratio = float(rig.bind_world[hips][1, 3]) / src_h if abs(src_h) > 1e-6 else 1.0
                    world = (src_p @ vals.reshape(-1, 3).T) * ratio
                    vals = (self.Pinv @ world).T.astype(np.float32)
            self.tracks.append((tgt, ch.path, ch.times, vals, ch.interp))

    @classmethod
    def from_tracks(cls, name, duration, tracks, rig):
        o = cls.__new__(cls)
        o.name, o.duration, o.tracks = name, max(duration, 1e-3), tracks
        hips = rig.roles.get("hips")
        o.hips = hips
        tp = rig.parent[hips] if hips is not None else -1
        o.P = rig.bind_world[tp][:3, :3].copy() if tp >= 0 else np.eye(3, dtype=np.float32)
        o.Pinv = np.linalg.inv(o.P).astype(np.float32)
        o.bind_wp = o.P @ rig.bind_t[hips] if hips is not None else None
        return o

    @staticmethod
    def _sample_track(times, vals, interp, t, path):
        k = int(np.searchsorted(times, t, side="right")) - 1
        if k < 0:
            return vals[0] if interp != "CUBICSPLINE" else vals[1]
        if k >= len(times) - 1:
            return vals[-1] if interp != "CUBICSPLINE" else vals[-2]
        t0, t1 = float(times[k]), float(times[k + 1])
        u = (t - t0) / max(t1 - t0, 1e-9)
        if interp == "STEP":
            return vals[k]
        if interp == "CUBICSPLINE":
            dt = t1 - t0
            p0, m0 = vals[3 * k + 1], vals[3 * k + 2] * dt
            p1, m1 = vals[3 * (k + 1) + 1], vals[3 * (k + 1)] * dt
            u2, u3 = u * u, u * u * u
            v = ((2 * u3 - 3 * u2 + 1) * p0 + (u3 - 2 * u2 + u) * m0 +
                 (-2 * u3 + 3 * u2) * p1 + (u3 - u2) * m1)
            return v / np.linalg.norm(v) if path == "rotation" else v
        a, b = vals[k], vals[k + 1]
        return slerp(a, b, u) if path == "rotation" else a + (b - a) * u

    def sample(self, rig: Rig, t: float, in_place=True):
        T, R, S = rig.bind_t.copy(), rig.bind_r.copy(), rig.bind_s.copy()
        for node, path, times, vals, interp in self.tracks:
            v = self._sample_track(times, vals, interp, t, path)
            if path == "translation":
                if in_place and node == self.hips:
                    wp = self.P @ v                       # lock horizontal root motion in WORLD space
                    wp[0], wp[2] = self.bind_wp[0], self.bind_wp[2]
                    v = self.Pinv @ wp
                T[node] = v
            elif path == "rotation":
                R[node] = v
            else:
                S[node] = v
        return T, R, S


class Animator:
    """Plays one clip at a time with cross-fades. Gets clips from any number of Mixamo-compatible
    GLB files (the model itself plus optional animation donors)."""

    def __init__(self, rig: Rig):
        self.rig = rig
        self.clips: Dict[str, RetargetedClip] = {}
        self.current: Optional[str] = None
        self.prev: Optional[str] = None
        self.time = 0.0
        self.prev_time = 0.0
        self.fade = 1.0
        self.fade_len = 0.3
        self.speed = 1.0
        self.loop = True
        self.paused = False
        self.finished = False

    def add_clips(self, model: GltfModel, tag: str = ""):
        """Add every clip of `model`. Clips of the rig's own model are used as they are; clips of
        another (Mixamo-compatible) model are baked onto this skeleton by world-space retargeting."""
        donor = None if model is self.rig.model else Rig(model)
        for c in model.clips:
            if c.duration <= 0.2:
                continue
            if donor is None:
                rc = RetargetedClip(c, self.rig.names, self.rig, None)
            else:
                rc = self._bake(donor, c)
            if rc is not None and rc.tracks:
                name = c.name if c.name not in self.clips else f"{tag}{c.name}"
                self.clips[name] = rc

    def _bake(self, dr: "Rig", clip: Clip, fps: float = 30.0) -> Optional[RetargetedClip]:
        rig = self.rig
        if "hips" not in rig.roles or "hips" not in dr.roles:
            return None
        src = RetargetedClip(clip, dr.names, dr, None)
        match = {}
        for ti, nm in enumerate(rig.names):
            j = dr.by_norm.get(norm_name(nm))
            if j is not None:
                match[ti] = j
        if len(match) < 20:
            return None
        # yaw that turns the donor's facing direction into ours
        fa, fb = dr.facing.copy(), rig.facing.copy()
        fa[1] = fb[1] = 0
        A = rotation_between(fa, fb)
        nrm = lambda m: m / np.maximum(np.linalg.norm(m, axis=0, keepdims=True), 1e-9)
        d_bind = np.array([nrm(dr.bind_world[i][:3, :3]) for i in range(dr.n)])
        t_bind = np.array([nrm(rig.bind_world[i][:3, :3]) for i in range(rig.n)])
        d_bind_inv = d_bind.transpose(0, 2, 1)
        hips_t, hips_d = rig.roles["hips"], dr.roles["hips"]
        ratio = float(rig.bind_world[hips_t][1, 3]) / float(dr.bind_world[hips_d][1, 3])
        d_bind_pos = dr.bind_world[hips_d][:3, 3].copy()
        tp = rig.parent[hips_t]
        P = rig.bind_world[tp][:3, :3] if tp >= 0 else np.eye(3, dtype=np.float32)
        Pinv = np.linalg.inv(P)
        parent_t = rig.bind_world[tp][:3, 3] if tp >= 0 else np.zeros(3, np.float32)
        n_frames = max(2, int(math.ceil(clip.duration * fps)) + 1)
        times = (np.arange(n_frames) / fps).astype(np.float32)
        quats = np.zeros((n_frames, rig.n, 4), np.float32)
        hips_loc = np.zeros((n_frames, 3), np.float32)
        bind_loc = np.array([quat_to_mat3(q) for q in rig.bind_r])
        for f, t in enumerate(times):
            T, R, S = src.sample(dr, min(float(t), clip.duration), in_place=False)
            dr.t, dr.r, dr.s = T, R, S
            dr.update_world()
            rw_s = np.array([nrm(dr.world[i][:3, :3]) for i in range(dr.n)])
            rw_t = np.zeros((rig.n, 3, 3), np.float32)
            for i in rig.order:
                p = rig.parent[i]
                rp = rw_t[p] if p >= 0 else np.eye(3, dtype=np.float32)
                if i in match:
                    j = match[i]
                    delta = rw_s[j] @ d_bind_inv[j]
                    rw_t[i] = (A @ delta @ A.T) @ t_bind[i]
                else:
                    rw_t[i] = rp @ bind_loc[i]
                q = mat3_to_quat(rp.T @ rw_t[i])
                if f > 0 and np.dot(q, quats[f - 1, i]) < 0:
                    q = -q
                quats[f, i] = q
            world = rig.bind_world[hips_t][:3, 3] + (A @ (dr.world[hips_d][:3, 3] - d_bind_pos)) * ratio
            hips_loc[f] = Pinv @ (world - parent_t)
        tracks = [(i, "rotation", times, quats[:, i], "LINEAR") for i in match if i != hips_t or True]
        tracks.append((hips_t, "translation", times, hips_loc, "LINEAR"))
        return RetargetedClip.from_tracks(clip.name, clip.duration, tracks, rig)

    def names(self):
        return list(self.clips)

    def find(self, *keywords) -> Optional[str]:
        for kw in keywords:
            for n in self.clips:
                if kw in n.lower():
                    return n
        return None

    def play(self, name, fade=0.3, speed=1.0, loop=True):
        if name not in self.clips:
            return False
        if name == self.current and not self.finished:
            self.loop, self.speed = loop, speed
            return True
        self.prev, self.prev_time = self.current, self.time
        self.current, self.time = name, 0.0
        self.fade, self.fade_len = 0.0, max(fade, 1e-3)
        self.speed, self.loop, self.finished = speed, loop, False
        return True

    def update(self, dt, in_place=True):
        rig = self.rig
        if self.current is None:
            return
        if not self.paused:
            self.time += dt * self.speed
            self.prev_time += dt * self.speed
            clip = self.clips[self.current]
            if self.time >= clip.duration:
                if self.loop:
                    self.time %= clip.duration
                else:
                    self.time = clip.duration - 1e-3
                    self.finished = True
        self.fade = min(1.0, self.fade + dt / self.fade_len)
        clip = self.clips[self.current]
        T, R, S = clip.sample(rig, self.time, in_place)
        if self.prev is not None and self.fade < 1.0:
            pc = self.clips[self.prev]
            T2, R2, S2 = pc.sample(rig, self.prev_time % pc.duration, in_place)
            f = self.fade * self.fade * (3 - 2 * self.fade)
            T, S = T2 + (T - T2) * f, S2 + (S - S2) * f
            R = slerp(R2, R, f)
        rig.t, rig.r, rig.s = T, R, S
        rig.update_world()