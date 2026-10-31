"""Build a smooth, articulated holographic humanoid without an external model."""
from __future__ import annotations

import math

import numpy as np

from .gltf_loader import GltfModel, Material, Node, Primitive
from .mathutil import quat_from_axis_angle


def _smooth_sphere(rings=16, segments=24):
    vertices = []
    indices = []
    for row in range(rings + 1):
        theta = math.pi * row / rings
        st, ct = math.sin(theta), math.cos(theta)
        for col in range(segments + 1):
            phi = 2.0 * math.pi * col / segments
            vertices.append((st * math.cos(phi), ct, st * math.sin(phi)))
    for row in range(rings):
        for col in range(segments):
            a = row * (segments + 1) + col
            b = a + segments + 1
            indices.extend((a, a + 1, b + 1, a, b + 1, b))
    vertices = np.asarray(vertices, np.float32)
    return vertices, vertices.copy(), np.asarray(indices, np.uint32)


def create_humanoid() -> GltfModel:
    """Create a human-proportioned robot with smooth armor and articulated joints."""
    sphere = _smooth_sphere()
    nodes = []
    primitives = []

    def add_node(name, parent=-1, position=(0, 0, 0), scale=(1, 1, 1),
                 mesh=False, color=0, rotation=None):
        index = len(nodes)
        nodes.append(Node(
            name=name,
            parent=parent,
            children=[],
            t=np.asarray(position, np.float32),
            r=np.asarray(rotation if rotation is not None else (0, 0, 0, 1), np.float32),
            s=np.asarray(scale, np.float32),
        ))
        if parent >= 0:
            nodes[parent].children.append(index)
        if mesh:
            positions, normals, indices = sphere
            primitives.append(Primitive(
                positions=positions,
                normals=normals,
                uvs=np.zeros((len(positions), 2), np.float32),
                indices=indices,
                joints=None,
                weights=None,
                material=color,
                node=index,
                name=name,
            ))
        return index

    def shell(parent, name, position, dimensions, color=0, rotation=None):
        return add_node(name, parent, position, dimensions, True, color, rotation)

    hips = add_node("mixamorig:Hips", position=(0, 0.88, 0))
    shell(hips, "pelvis armor", (0, -0.015, 0), (0.18, 0.12, 0.13), 1)
    shell(hips, "waist reactor", (0, -0.015, 0.105), (0.075, 0.075, 0.025), 2)

    spine = add_node("mixamorig:Spine", hips, position=(0, 0.12, 0))
    shell(spine, "abdomen", (0, 0.105, 0), (0.18, 0.20, 0.105), 0)
    for y in (0.04, 0.10, 0.16):
        shell(spine, "abdominal rib", (0, y, 0.097), (0.145, 0.018, 0.02), 1)

    chest = add_node("mixamorig:Spine2", spine, position=(0, 0.23, 0))
    shell(chest, "chest shell", (0, 0.055, 0), (0.265, 0.19, 0.125), 1)
    shell(chest, "chest core housing", (0, 0.065, 0.112), (0.085, 0.085, 0.035), 3)
    shell(chest, "chest power core", (0, 0.065, 0.14), (0.046, 0.046, 0.018), 2)
    for sign in (-1, 1):
        shell(chest, "pectoral plate", (sign * 0.15, 0.09, 0.047),
              (0.09, 0.105, 0.082), 0)
        shell(chest, "shoulder light", (sign * 0.232, 0.17, 0.015),
              (0.045, 0.028, 0.04), 2)

    neck = add_node("mixamorig:Neck", chest, position=(0, 0.19, 0))
    shell(neck, "neck joint", (0, 0.025, 0), (0.075, 0.09, 0.075), 3)
    head = add_node("mixamorig:Head", neck, position=(0, 0.105, 0))
    shell(head, "helmet", (0, 0.065, 0), (0.14, 0.17, 0.13), 1)
    shell(head, "face visor", (0, 0.05, 0.112), (0.112, 0.067, 0.03), 3)
    for sign in (-1, 1):
        shell(head, "optic", (sign * 0.044, 0.056, 0.14),
              (0.027, 0.018, 0.014), 2)
        shell(head, "ear actuator", (sign * 0.145, 0.045, 0),
              (0.035, 0.055, 0.065), 0)
    shell(head, "forehead sensor", (0, 0.155, 0.105), (0.025, 0.035, 0.02), 2)

    for side, sign in (("Left", 1.0), ("Right", -1.0)):
        arm_rotation = quat_from_axis_angle((0, 0, 1), -sign * 0.48)
        shoulder = add_node(
            f"mixamorig:{side}Arm", chest,
            position=(sign * 0.255, 0.115, 0), rotation=arm_rotation,
        )
        shell(shoulder, "shoulder actuator", (sign * 0.045, 0, 0),
              (0.105, 0.115, 0.12), 3)
        shell(shoulder, "upper arm armor", (sign * 0.17, -0.008, 0),
              (0.19, 0.075, 0.075), 1)
        shell(shoulder, "upper arm light", (sign * 0.17, -0.008, 0.065),
              (0.12, 0.018, 0.012), 2)

        elbow = add_node(
            f"mixamorig:{side}ForeArm", shoulder,
            position=(sign * 0.29, -0.12, 0),
        )
        shell(elbow, "elbow joint", (0, 0, 0), (0.073, 0.078, 0.078), 3)
        shell(elbow, "forearm armor", (sign * 0.15, 0, 0),
              (0.18, 0.065, 0.065), 0)
        shell(elbow, "forearm light", (sign * 0.15, 0, 0.058),
              (0.10, 0.014, 0.01), 2)

        hand = add_node(
            f"mixamorig:{side}Hand", elbow,
            position=(sign * 0.27, -0.045, 0),
        )
        shell(hand, "palm", (sign * 0.045, 0, 0),
              (0.09, 0.06, 0.075), 1)
        finger_names = ("Thumb", "Index", "Middle", "Ring", "Pinky")
        finger_offsets = (0.050, 0.026, 0.0, -0.026, -0.050)
        for finger_name, offset in zip(finger_names, finger_offsets):
            parent = hand
            start = (sign * 0.095, offset, 0)
            for segment in range(1, 4):
                joint = add_node(
                    f"mixamorig:{side}Hand{finger_name}{segment}",
                    parent,
                    position=start if segment == 1 else (sign * 0.035, 0, 0),
                )
                shell(joint, f"{side} {finger_name} segment {segment}",
                      (sign * 0.021, 0, 0),
                      (0.038, 0.012, 0.013), 2 if segment == 3 else 0)
                parent = joint

        hip = add_node(
            f"mixamorig:{side}UpLeg", hips,
            position=(sign * 0.105, -0.09, 0),
        )
        shell(hip, "hip actuator", (0, -0.025, 0),
              (0.105, 0.115, 0.11), 3)
        shell(hip, "thigh armor", (0, -0.17, 0),
              (0.09, 0.19, 0.095), 1)
        shell(hip, "thigh light", (0, -0.17, 0.084),
              (0.035, 0.10, 0.015), 2)

        knee = add_node(
            f"mixamorig:{side}Leg", hip,
            position=(0, -0.34, 0),
        )
        shell(knee, "knee joint", (0, 0, 0.025),
              (0.075, 0.078, 0.055), 3)
        shell(knee, "shin armor", (0, -0.17, 0),
              (0.07, 0.19, 0.075), 0)
        shell(knee, "shin light", (0, -0.17, 0.066),
              (0.028, 0.09, 0.012), 2)

        foot = add_node(
            f"mixamorig:{side}Foot", knee,
            position=(0, -0.34, 0),
        )
        shell(foot, "ankle", (0, 0.01, 0),
              (0.06, 0.075, 0.06), 3)
        shell(foot, "foot shell", (0, -0.015, 0.065),
              (0.085, 0.055, 0.13), 1)

    materials = [
        Material(base_color=np.array([0.82, 0.88, 0.94, 1.0], np.float32), name="ceramic alloy"),
        Material(base_color=np.array([0.96, 0.98, 1.0, 1.0], np.float32), name="white armor"),
        Material(base_color=np.array([0.10, 0.78, 1.0, 1.0], np.float32), name="cyan indicators"),
        Material(base_color=np.array([0.018, 0.028, 0.045, 1.0], np.float32), name="dark mechanisms"),
    ]
    return GltfModel(
        nodes=nodes,
        roots=[hips],
        primitives=primitives,
        skins=[],
        clips=[],
        materials=materials,
        images=[],
        path="<procedural humanoid>",
    )
