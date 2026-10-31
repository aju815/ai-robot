"""Minimal GLB (glTF 2.0 binary) loader.

Turns a .glb file into plain numpy data (`GltfModel`):
  * node hierarchy with local translation / rotation / scale
  * triangle meshes (positions, normals, uvs, joints, weights, indices)
  * skins (joint list + inverse bind matrices)
  * animations (LINEAR / STEP / CUBICSPLINE samplers)
  * materials (base colour factor, base colour texture, alpha mode) + decoded textures
No third-party glTF library is needed: only numpy and OpenCV (for image decoding).
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np

_COMPONENT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16,
              5125: np.uint32, 5126: np.float32}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}


class GltfError(RuntimeError):
    pass


# ---------------------------------------------------------------- data model
@dataclass
class Material:
    base_color: np.ndarray = field(default_factory=lambda: np.ones(4, np.float32))
    texture: Optional[int] = None          # index into GltfModel.images
    alpha_mode: str = "OPAQUE"             # OPAQUE | MASK | BLEND
    alpha_cutoff: float = 0.5
    double_sided: bool = False
    name: str = ""


@dataclass
class Primitive:
    positions: np.ndarray                  # (N,3) float32
    normals: np.ndarray                    # (N,3) float32
    uvs: np.ndarray                        # (N,2) float32
    indices: np.ndarray                    # (M,) uint32
    joints: Optional[np.ndarray]           # (N,4) int32   (None if not skinned)
    weights: Optional[np.ndarray]          # (N,4) float32
    material: int                          # index into GltfModel.materials (-1 = default)
    node: int                              # node that owns the mesh
    name: str = ""


@dataclass
class Skin:
    joints: List[int]                      # node indices
    inverse_bind: np.ndarray               # (J,4,4) float32, column-vector convention


@dataclass
class Channel:
    node: int
    path: str                              # translation | rotation | scale | weights
    times: np.ndarray                      # (K,)
    values: np.ndarray                     # (K,C) or (K*3,C) for cubic spline
    interp: str                            # LINEAR | STEP | CUBICSPLINE


@dataclass
class Clip:
    name: str
    channels: List[Channel]
    duration: float


@dataclass
class Node:
    name: str
    parent: int
    children: List[int]
    t: np.ndarray                          # (3,)
    r: np.ndarray                          # (4,) quaternion x,y,z,w
    s: np.ndarray                          # (3,)
    mesh: Optional[int] = None
    skin: Optional[int] = None


@dataclass
class GltfModel:
    nodes: List[Node]
    roots: List[int]
    primitives: List[Primitive]
    skins: List[Skin]
    clips: List[Clip]
    materials: List[Material]
    images: List[Optional[np.ndarray]]     # decoded RGBA uint8 (H,W,4) or None
    path: str = ""


# ---------------------------------------------------------------- helpers
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
        s = math_sqrt(tr + 1.0) * 2
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math_sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math_sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = math_sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    q = np.array(q, np.float32)
    return q / (np.linalg.norm(q) + 1e-12)


def math_sqrt(v):
    return float(np.sqrt(v))


def decompose(m4):
    """4x4 column-vector matrix -> (t, quat, s)."""
    t = m4[:3, 3].copy()
    sx, sy, sz = (np.linalg.norm(m4[:3, i]) for i in range(3))
    rot = m4[:3, :3] / np.array([sx or 1, sy or 1, sz or 1], np.float32)
    return t.astype(np.float32), mat3_to_quat(rot), np.array([sx, sy, sz], np.float32)


# ---------------------------------------------------------------- parser
class _Reader:
    def __init__(self, doc, blob):
        self.doc, self.blob = doc, blob

    def view_bytes(self, vi):
        bv = self.doc["bufferViews"][vi]
        if bv.get("buffer", 0) != 0:
            raise GltfError("external buffers are not supported (use a .glb file)")
        off = bv.get("byteOffset", 0)
        return self.blob[off: off + bv["byteLength"]], bv.get("byteStride")

    def accessor(self, ai):
        a = self.doc["accessors"][ai]
        dtype = np.dtype(_COMPONENT[a["componentType"]])
        ncomp = _NCOMP[a["type"]]
        count = a["count"]
        if "bufferView" in a:
            raw, stride = self.view_bytes(a["bufferView"])
            off = a.get("byteOffset", 0)
            esize = dtype.itemsize * ncomp
            stride = stride or esize
            arr = np.ndarray((count, ncomp), dtype, raw, off, (stride, dtype.itemsize)).copy()
        else:
            arr = np.zeros((count, ncomp), dtype)
        if "sparse" in a:
            sp = a["sparse"]
            idx_raw, _ = self.view_bytes(sp["indices"]["bufferView"])
            idt = np.dtype(_COMPONENT[sp["indices"]["componentType"]])
            idx = np.frombuffer(idx_raw, idt, sp["count"], sp["indices"].get("byteOffset", 0))
            val_raw, _ = self.view_bytes(sp["values"]["bufferView"])
            vals = np.frombuffer(val_raw, dtype, sp["count"] * ncomp,
                                 sp["values"].get("byteOffset", 0)).reshape(-1, ncomp)
            arr[idx] = vals
        if a.get("normalized") and dtype.kind in "iu":
            info = np.iinfo(dtype)
            arr = np.maximum(arr.astype(np.float32) / float(info.max), -1.0)
        return arr


def load_glb(path: str) -> GltfModel:
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 20 or data[:4] != b"glTF":
        raise GltfError(f"{path}: not a GLB file")
    version, _length = struct.unpack_from("<II", data, 4)
    if version != 2:
        raise GltfError(f"{path}: unsupported glTF version {version}")
    pos, doc, blob = 12, None, b""
    while pos + 8 <= len(data):
        clen, ctype = struct.unpack_from("<II", data, pos)
        chunk = data[pos + 8: pos + 8 + clen]
        if ctype == 0x4E4F534A:
            doc = json.loads(chunk.decode("utf-8"))
        elif ctype == 0x004E4942:
            blob = chunk
        pos += 8 + clen
    if doc is None:
        raise GltfError(f"{path}: missing JSON chunk")
    rd = _Reader(doc, blob)

    # ---- images (decoded to RGBA)
    images: List[Optional[np.ndarray]] = []
    for im in doc.get("images", []):
        img = None
        if "bufferView" in im:
            raw, _ = rd.view_bytes(im["bufferView"])
            buf = np.frombuffer(raw, np.uint8)
            dec = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
            if dec is not None:
                if dec.ndim == 2:
                    dec = cv2.cvtColor(dec, cv2.COLOR_GRAY2RGBA)
                elif dec.shape[2] == 3:
                    dec = cv2.cvtColor(dec, cv2.COLOR_BGR2RGBA)
                else:
                    dec = cv2.cvtColor(dec, cv2.COLOR_BGRA2RGBA)
                img = np.ascontiguousarray(dec)
        images.append(img)
    tex_to_image = [t.get("source") for t in doc.get("textures", [])]

    # ---- materials
    materials: List[Material] = []
    for m in doc.get("materials", []):
        pbr = m.get("pbrMetallicRoughness", {})
        bt = pbr.get("baseColorTexture")
        tex = tex_to_image[bt["index"]] if bt is not None else None
        materials.append(Material(
            base_color=np.array(pbr.get("baseColorFactor", [1, 1, 1, 1]), np.float32),
            texture=tex, alpha_mode=m.get("alphaMode", "OPAQUE"),
            alpha_cutoff=float(m.get("alphaCutoff", 0.5)),
            double_sided=bool(m.get("doubleSided", False)), name=m.get("name", "")))

    # ---- nodes
    jnodes = doc.get("nodes", [])
    nodes: List[Node] = []
    for i, n in enumerate(jnodes):
        if "matrix" in n:
            m4 = np.array(n["matrix"], np.float32).reshape(4, 4).T
            t, r, s = decompose(m4)
        else:
            t = np.array(n.get("translation", [0, 0, 0]), np.float32)
            r = np.array(n.get("rotation", [0, 0, 0, 1]), np.float32)
            r = r / (np.linalg.norm(r) + 1e-12)
            s = np.array(n.get("scale", [1, 1, 1]), np.float32)
        nodes.append(Node(n.get("name", f"node{i}"), -1, list(n.get("children", [])), t, r, s,
                          n.get("mesh"), n.get("skin")))
    for i, n in enumerate(nodes):
        for c in n.children:
            nodes[c].parent = i
    scene = doc.get("scenes", [{}])[doc.get("scene", 0)]
    roots = list(scene.get("nodes", [i for i, n in enumerate(nodes) if n.parent < 0]))

    # ---- skins
    skins: List[Skin] = []
    for sk in doc.get("skins", []):
        joints = list(sk["joints"])
        if "inverseBindMatrices" in sk:
            ibm = rd.accessor(sk["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
        else:
            ibm = np.tile(np.eye(4, dtype=np.float32), (len(joints), 1, 1))
        skins.append(Skin(joints, np.ascontiguousarray(ibm, np.float32)))

    # ---- meshes -> primitives (one entry per node that uses the mesh)
    prims: List[Primitive] = []
    for ni, n in enumerate(nodes):
        if n.mesh is None:
            continue
        mesh = doc["meshes"][n.mesh]
        for pj, p in enumerate(mesh["primitives"]):
            if p.get("mode", 4) != 4:
                continue                                    # only triangle lists
            at = p["attributes"]
            if "POSITION" not in at:
                continue
            pos = rd.accessor(at["POSITION"]).astype(np.float32)
            nvert = len(pos)
            if "indices" in p:
                idx = rd.accessor(p["indices"]).reshape(-1).astype(np.uint32)
            else:
                idx = np.arange(nvert, dtype=np.uint32)
            idx = idx[: len(idx) // 3 * 3]
            if "NORMAL" in at:
                nor = rd.accessor(at["NORMAL"]).astype(np.float32)
            else:
                nor = _compute_normals(pos, idx)
            uv = (rd.accessor(at["TEXCOORD_0"]).astype(np.float32)
                  if "TEXCOORD_0" in at else np.zeros((nvert, 2), np.float32))
            joints = weights = None
            if "JOINTS_0" in at and "WEIGHTS_0" in at and n.skin is not None:
                joints = rd.accessor(at["JOINTS_0"]).astype(np.int32)
                weights = rd.accessor(at["WEIGHTS_0"]).astype(np.float32)
                s = weights.sum(1, keepdims=True)
                weights = np.where(s > 1e-8, weights / np.maximum(s, 1e-8), weights)
            prims.append(Primitive(pos, nor, uv, idx, joints, weights,
                                   p.get("material", -1), ni, f"{mesh.get('name', 'mesh')}#{pj}"))

    # ---- animations
    clips: List[Clip] = []
    for ai, an in enumerate(doc.get("animations", [])):
        chans: List[Channel] = []
        dur = 0.0
        for ch in an["channels"]:
            tgt = ch["target"]
            if "node" not in tgt or tgt["path"] not in ("translation", "rotation", "scale"):
                continue
            smp = an["samplers"][ch["sampler"]]
            times = rd.accessor(smp["input"]).reshape(-1).astype(np.float32)
            vals = rd.accessor(smp["output"]).astype(np.float32)
            interp = smp.get("interpolation", "LINEAR")
            if len(times) == 0:
                continue
            dur = max(dur, float(times[-1]))
            chans.append(Channel(tgt["node"], tgt["path"], times, vals, interp))
        clips.append(Clip(an.get("name", f"clip{ai}"), chans, dur))

    return GltfModel(nodes, roots, prims, skins, clips, materials, images, path)


def _compute_normals(pos, idx):
    tri = idx.reshape(-1, 3)
    a, b, c = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
    fn = np.cross(b - a, c - a)
    nor = np.zeros_like(pos)
    for k in range(3):
        np.add.at(nor, tri[:, k], fn)
    l = np.linalg.norm(nor, axis=1, keepdims=True)
    return (nor / np.maximum(l, 1e-12)).astype(np.float32)