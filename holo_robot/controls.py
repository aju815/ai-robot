"""SAM-X1-style gesture controls for the holographic humanoid."""
from __future__ import annotations

import collections
import math
from typing import Optional

import numpy as np

from .mathutil import rot_axis_angle_mat3
from .rig import Rig
from .scene import Scene


MODES = {
    0: ("STANDBY", "Fist: power down; open fingers to wake."),
    1: ("HEAD UNIT", "1 finger: move hand left/right to look, up/down to nod."),
    2: ("TORSO / CORE", "2 fingers: lean left/right; move up/down to adjust core."),
    3: ("ARM SYSTEM", "3 fingers: move up/down to raise arms; left/right to flex elbows."),
    4: ("LEG SYSTEM", "4 fingers: move left/right to spread legs; up/down to crouch."),
    5: ("FULL BODY SYNC", "5 fingers: move your hand to drive the whole robot."),
}

DEFAULT = {
    "yaw": 0.0, "tilt": 0.0, "lean": 0.0, "core": 0.55,
    "raise": 0.0, "elbow": 0.0, "spread": 0.0, "crouch": 0.0,
}
STANDBY = {
    "yaw": 0.0, "tilt": 0.5, "lean": 0.04, "core": 0.04,
    "raise": 0.0, "elbow": 0.0, "spread": 0.0, "crouch": 0.18,
}

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


def _distance(a, b) -> float:
    return float(np.hypot(a.x - b.x, a.y - b.y))


def count_fingers(landmarks) -> int:
    """Count raised fingers using palm-relative landmark distances."""
    count = int(_distance(landmarks[4], landmarks[17]) >
                _distance(landmarks[3], landmarks[17]))
    for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
        if _distance(landmarks[tip], landmarks[0]) > \
                _distance(landmarks[pip], landmarks[0]) * 1.05:
            count += 1
    return count


def finger_curls(landmarks) -> np.ndarray:
    """Estimate thumb-to-pinky curl from the bend at each finger's middle joint."""
    joints = ((2, 3, 4), (5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20))
    curls = []
    for proximal, middle, tip in joints:
        a = np.array([landmarks[proximal].x - landmarks[middle].x,
                      landmarks[proximal].y - landmarks[middle].y], np.float32)
        b = np.array([landmarks[tip].x - landmarks[middle].x,
                      landmarks[tip].y - landmarks[middle].y], np.float32)
        denominator = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-8)
        angle = math.acos(float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0)))
        curls.append(float(np.clip(1.0 - angle / math.pi, 0.0, 1.0)))
    return np.asarray(curls, np.float32)


def hand_from_landmarks(landmarks, smooth: dict) -> dict:
    """Convert 21 MediaPipe landmarks into SAM-compatible gesture and pointer data."""
    palm = landmarks[9]
    for key, value in (("x", palm.x), ("y", palm.y),
                       ("pointer_x", landmarks[8].x), ("pointer_y", landmarks[8].y)):
        previous = smooth.get(key)
        amount = 0.4 if key in ("x", "y") else 0.45
        smooth[key] = value if previous is None else previous * (1.0 - amount) + value * amount

    u = float(np.clip((smooth["x"] - 0.5) / 0.28, -1.0, 1.0))
    v = float(np.clip((0.5 - smooth["y"]) / 0.28, -1.0, 1.0))
    palm_width = max(_distance(landmarks[0], landmarks[9]), 1e-6)
    pinch = _distance(landmarks[4], landmarks[8]) / palm_width < 0.38
    return {
        "count": count_fingers(landmarks),
        "u": u,
        "v": v,
        "x": smooth["x"],
        "y": smooth["y"],
        "pointer_x": float(np.clip(smooth["pointer_x"], 0.0, 1.0)),
        "pointer_y": float(np.clip(smooth["pointer_y"], 0.0, 1.0)),
        "pinch": pinch,
        "curls": finger_curls(landmarks),
    }


def update_two_hand_zoom(hands: dict, camera, previous_span: Optional[float]) -> Optional[float]:
    """Zoom with both hands pinched; spreading apart zooms in."""
    left, right = hands.get("left"), hands.get("right")
    if left is None or right is None or not left["pinch"] or not right["pinch"]:
        return None

    span = math.hypot(
        (left["x"] - right["x"]) * camera.w,
        (left["y"] - right["y"]) * camera.h,
    )
    if previous_span is not None and span > 1.0:
        camera.zoom(float(np.clip(previous_span / span, 0.96, 1.04)))
    return span


class SamGestureController:
    """Stable finger-count modes, pinch selection, and smoothed SAM robot controls."""

    def __init__(self):
        self.mode = 0
        self.values = dict(STANDBY)
        self.targets = dict(STANDBY)
        self.history = collections.deque(maxlen=6)
        self.pinch_was_active = False
        self.hand: Optional[dict] = None
        self.pointer: Optional[tuple[int, int]] = None
        self.hover = None
        self.selected_region = -1
        self.selected_part = "HOLOGRAM ROBOT"

    @property
    def mode_name(self) -> str:
        return MODES[self.mode][0]

    @property
    def instructions(self) -> str:
        return MODES[self.mode][1]

    def update(self, hand: Optional[dict], scene: Scene, elapsed: float) -> None:
        self.hand = hand
        if hand is None:
            self.history.clear()
            self.pointer = None
            self.hover = None
            self.pinch_was_active = False
        else:
            self.pointer = (
                int(hand["pointer_x"] * scene.camera.w),
                int(hand["pointer_y"] * scene.camera.h),
            )
            self.hover = scene.pick(*self.pointer)
            pinching = bool(hand["pinch"])
            if pinching and not self.pinch_was_active and self.hover is not None:
                kind, target, _ = self.hover
                if kind == "region":
                    region_name = scene.regions[target][0]
                    self.selected_region = target
                    self.selected_part = region_name
                    if "HEAD UNIT" in region_name:
                        self._set_mode(1)
                    elif "TORSO CORE" in region_name:
                        self._set_mode(2)
                    elif "ARM" in region_name:
                        self._set_mode(3)
                    elif "LEG" in region_name:
                        self._set_mode(4)
                elif kind == "object":
                    target.selected = not target.selected
            self.pinch_was_active = pinching

            if not pinching:
                self.history.append(hand["count"])
                if len(self.history) == self.history.maxlen and \
                        len(set(self.history)) == 1 and self.history[0] != self.mode:
                    self._set_mode(self.history[0])

            u, v = hand["u"], hand["v"]
            if self.mode == 1:
                self.targets["yaw"], self.targets["tilt"] = u, -v * 0.35
            elif self.mode == 2:
                self.targets["lean"], self.targets["core"] = u * 0.35, (v + 1.0) / 2.0
            elif self.mode == 3:
                self.targets["raise"] = (v + 1.0) / 2.0 * (math.pi / 2.0)
                self.targets["elbow"] = (u + 1.0) / 2.0 * (math.pi / 2.0)
            elif self.mode == 4:
                self.targets["spread"] = (u + 1.0) / 2.0 * 0.6
                self.targets["crouch"] = (1.0 - v) / 2.0 * 0.9
            elif self.mode == 5:
                amount = (v + 1.0) / 2.0
                self.targets["raise"] = amount * 1.4
                self.targets["spread"] = amount * 0.5
                self.targets["lean"], self.targets["yaw"] = u * 0.25, u
                self.targets["tilt"] = -v * 0.35
                self.targets["crouch"] = (1.0 - v) / 2.0 * 0.45
                self.targets["elbow"] = 0.5 + 0.35 * math.sin(elapsed * 4.0)
                self.targets["core"] = 0.4 + 0.6 * amount

        for key in self.values:
            self.values[key] += (self.targets[key] - self.values[key]) * 0.2

    def _set_mode(self, mode: int) -> None:
        was_standby = self.mode == 0
        self.mode = mode
        if mode == 0:
            self.targets = dict(STANDBY)
        elif was_standby:
            self.targets = dict(DEFAULT)

    def apply_pose(self, rig: Rig) -> None:
        if self.mode == 0:
            return
        p = self.values

        def rotate(role: str, axis, angle: float) -> None:
            index = rig.roles.get(role)
            if index is not None and abs(angle) > 1e-5:
                rig._apply_world_rotation(
                    index, rot_axis_angle_mat3(axis, angle)
                )

        rotate("spine", (0, 0, 1), p["lean"] * 0.45)
        rotate("chest", (0, 0, 1), p["lean"] * 0.55)
        rotate("head", (0, 1, 0), p["yaw"] * 0.75)
        rotate("neck", (0, 1, 0), p["yaw"] * 0.25)
        rotate("head", (1, 0, 0), p["tilt"])

        for side, sign in (("left", -1.0), ("right", 1.0)):
            rotate(f"{side}_arm", (0, 0, 1), sign * p["raise"])
            rotate(f"{side}_forearm", (0, 0, 1), -sign * p["elbow"])
            rotate(f"{side}_upleg", (0, 0, 1), sign * p["spread"])
            rotate(f"{side}_leg", (1, 0, 0), -p["crouch"])

        rig.update_world()
