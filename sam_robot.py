#!/usr/bin/env python3
"""
SHANIB AJU ROBOT 3D  v16.0  -  Intelligent Robotics Studio with VOICE
=============================================================================
Same robot, same head / hand / face tracking, same voice and AI brain as v13.1 -
with a more focused, professional studio interface:

    * the robot remains a translucent GLASS HOLOGRAM with a crisp cyan wireframe,
      hidden back edges visible (x-ray), and selected parts highlighted in amber
    * a deep navy studio, soft light pillar, restrained projector rings,
      rotating segmented rings, orbiting energy orbs with orbit paths
    * a structured control-room layout with live tracking, motion telemetry, AI guidance,
      voice interaction, and a compact shortcut bar
    * clean glass panels, subtle borders, readable labels, and a consistent cyan-and-amber accent

TALK TO THE ROBOT
    just speak: your words appear as text (YOU SAID) and the robot answers in a speech
    bubble next to its head, moves its mouth and speaks the answer out loud.
    Y = type to the robot (Enter sends, Esc cancels)   M = mute mic   X = stop talking
    voice or typed commands: sit / stand / lift your leg / jump / push-up /
                     jumping jacks / exercise / march in place / yoga / tree pose /
                     warrior pose / meditation / squat / stretch /
                     dance / raise your hand / wave / rotate your head 360 / stop head rotation /
                     lower your hand / stop dancing / explode / assemble / zoom in / zoom out /
                     rotate left / turn around / spin / stop spinning / stand by / wake up /
                     take a screenshot / ask my name or version / "show the knee" / ...
                     or ask anything (needs Ollama)

YOUR HEAD CONTROLS THE ROBOT HEAD   (C = recalibrate)
HANDS: each hand moves the arm on its side, fingers curl like yours
    pinch on a part = select + AI guide      pinch in empty space + move = rotate 360
    pinch with BOTH hands = zoom             clap = EXPLODE / assemble
    both hands as fists = STANDBY (open a hand to wake)

MOUSE / KEYS
    drag = rotate 360      wheel or + / - = zoom     0 or K = full robot
    E explode / assemble   F zoom on selected part   T auto-spin
    J / L turn left/right  N / B next / previous     I ask the AI again
    U fullscreen/windowed             V webcam: box -> full -> hidden
    H overlay on/off   C calibrate head
    R rotate camera 180    S screenshot              Q / ESC quit

AI BRAIN (optional)
        ollama serve ; ollama pull qwen3:8b
        python3 sam_robot.py --lang english --ollama-model qwen3:8b
        python3 sam_robot.py --list-voices

SETUP
    python3 -m pip install opencv-python mediapipe numpy Pillow SpeechRecognition pyaudio pyttsx3
    macOS native Unicode text: python3 -m pip install pyobjc-framework-Cocoa
    python3 sam_robot.py | --no-camera | --no-voice | --no-speak
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from typing import Optional
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    if sys.platform == "darwin":
        from AppKit import (NSAttributedString, NSBitmapImageRep, NSDeviceRGBColorSpace,
                            NSFont, NSFontAttributeName, NSForegroundColorAttributeName,
                            NSGraphicsContext, NSMakePoint, NSColor)
        from Foundation import NSString
    else:
        NSBitmapImageRep = None
except ImportError:
    NSBitmapImageRep = None

APP_VERSION = "16.0"
ROBOT_NAME = "Shanib Aju"
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_D = cv2.FONT_HERSHEY_DUPLEX
AA = cv2.LINE_AA

# ---- studio palette (BGR): crisp cyan, warm amber, and deep navy
CYAN = (232, 194, 88)
CYAN_DIM = (116, 101, 58)
CYAN_DARK = (53, 43, 27)
WHITE = (255, 255, 255)
ORANGE = (70, 165, 255)
GOOD = (130, 225, 120)
GREY = (190, 190, 190)
RED = (75, 70, 245)

INK1 = (247, 249, 252)       # primary text
INK2 = (198, 211, 226)       # secondary text
INK3 = (135, 154, 176)       # tertiary / labels
HAIR = (52, 65, 82)          # hairline rules
HAIR2 = (82, 108, 132)       # stronger hairline

HAND_CONN = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
             (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
             (18, 19), (19, 20), (0, 17)]

# materials: plastic, joints, eyes, metal, hair, shirt, trousers, and fabric trim
MAT_COL = np.array([
    [244, 244, 246], [46, 46, 52], [255, 232, 150], [150, 150, 158],
    [30, 39, 58], [170, 75, 38], [72, 58, 43], [110, 142, 194],
], np.float32)  # BGR (kept for reference)
# hologram glass colours per material (BGR)
HOLO_COL = np.array([
    [238, 245, 250], [112, 91, 70], [245, 252, 255], [188, 202, 220],
    [125, 151, 179], [206, 147, 94], [116, 108, 87], [238, 244, 251],
], np.float32)
MAT_SPEC = np.array([0.60, 0.40, 0.0, 0.30, 0.12, 0.08, 0.06, 0.10], np.float32)
WHITE_M, BLACK_M, EYE_M, METAL_M, HAIR_M, SHIRT_M, PANTS_M, FABRIC_TRIM_M = range(8)

# --------------------------------------------------------------------------- part notes
PART_INFO = {
    "HEAD": "Sensor head with stereo cameras behind the eyes, microphones in the black ear pods, "
            "a smooth white shell and dark brown hair. It handles vision, speech and gaze. It copies "
            "the movement of your own head and talks to you.",
    "NECK": "Black ribbed three-axis neck joint. Small servos yaw, nod and tilt the head, and a "
            "slip ring routes camera and power cables through it without twisting.",
    "CHEST": "Layered chest armour under a tailored blue shirt with a collar, buttons and stitched "
             "details. It protects the battery pack, the main computer and the motor drivers, and it "
             "carries the shoulders.",
    "ABDOMEN": "Flexible black waist with ribbed bellows. It lets the upper body lean over the "
               "pelvis while power and data cables pass through.",
    "PELVIS": "White hip frame that carries the whole robot. It holds the hip actuators and the "
              "balance sensors (IMU) that keep the body upright.",
    "SHOULDER": "Ball-style black shoulder joint with a large servo and a white cover pod. It lifts "
                "and swings the arm and carries the weight of everything below it.",
    "UPPER ARM": "Light shell around the shoulder-to-elbow link. It holds cables and part of the "
                 "actuator that drives the elbow.",
    "ELBOW": "Single-axis hinge with a high-torque servo and an angle sensor. It bends the "
             "forearm; here it is solved from your hand position.",
    "FOREARM": "Forearm link holding the wrist actuators and the finger motors. The black cuff "
               "at the end is the wrist joint.",
    "HAND": "Five-finger hand with tendon-driven fingers, black knuckle joints and fingertip touch "
            "sensors. Each finger curls with yours from the camera tracking.",
    "HIP": "Large black ball hip joint with a strong servo. It swings the leg forward, back and "
           "to the side while carrying the body weight.",
    "THIGH": "Upper leg link with a black inner panel, the strong hip and knee actuators. It carries "
             "the body weight when walking or crouching.",
    "KNEE": "Black hinge housing with a large servo, a side disc and a spring element. It absorbs "
            "impact and folds the leg for crouching and walking.",
    "SHIN": "Lower leg link with a black upper panel. It holds the ankle actuators and the cabling "
            "to the foot sensors, kept light so the leg can swing quickly.",
    "FOOT": "White shoe-shaped foot with pressure sensors under the sole and a black ankle joint. "
            "They measure where the weight is so the balance controller keeps the robot stable.",
}


def kind_of(name: str) -> str:
    return name.replace("LEFT ", "").replace("RIGHT ", "").strip()


# --------------------------------------------------------------------------- AI guide
class OllamaGuide:
    """Explains the selected part with a local Ollama model (background thread)."""

    def __init__(self, url: str, model: str, enabled: bool = True):
        self.url = url.rstrip("/")
        self.model = model
        self.enabled = enabled
        self.cache: dict[str, str] = {}
        self.part = ""
        self.text = ""
        self.status = "AI guide off (--offline)" if not enabled else "connecting to Ollama..."
        self._token = 0
        if enabled:
            threading.Thread(target=self._probe, daemon=True).start()

    def _probe(self):
        try:
            with urllib.request.urlopen(self.url + "/api/tags", timeout=3) as r:
                names = [m["name"] for m in json.load(r).get("models", [])]
            if names and self.model not in names:
                chat = [n for n in names if "embed" not in n.lower()]
                if chat:
                    self.model = chat[0]
            self.status = f"Ollama ready: {self.model}" if names else "Ollama has no models installed"
        except Exception:
            self.status = "Ollama offline - showing built-in notes"

    def request(self, part: str, force: bool = False):
        self.part = part
        self._token += 1
        token = self._token
        if not force and part in self.cache:
            self.text = self.cache[part]
            self.status = f"AI explanation (cached, {self.model})"
            return
        self.text = PART_INFO.get(kind_of(part), "No notes for this part.")
        if not self.enabled:
            self.status = "Built-in notes (AI guide off)"
            return
        self.status = f"asking {self.model}..."
        threading.Thread(target=self._ask, args=(part, token), daemon=True).start()

    def _ask(self, part: str, token: int):
        prompt = (
            "You are the guide of a white humanoid robot with black articulated joints, electric "
            "servo actuators, cameras and balance sensors. Explain the robot's "
            f"'{part.lower()}' to a curious visitor: what it is, how it moves or what it senses, "
            "and why it matters. Maximum 70 words. Plain text only, no markdown, no lists."
        )
        body = json.dumps({"model": self.model, "prompt": prompt, "stream": False, "think": False,
                           "options": {"temperature": 0.4, "num_predict": 170}}).encode()
        req = urllib.request.Request(self.url + "/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                out = json.load(r).get("response", "").strip()
            out = out.replace("**", "").replace("#", "").replace("`", "").strip()
            if not out:
                raise ValueError("empty answer")
            self.cache[part] = out
            if token == self._token:
                self.text = out
                self.status = f"AI explanation ({self.model})"
        except Exception as exc:
            if token == self._token:
                self.status = f"Ollama explanation failed ({type(exc).__name__}); built-in notes shown"


# --------------------------------------------------------------------------- 3D maths + meshes
def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], np.float32)


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float32)


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)


def v3(*a):
    return np.array(a, np.float32)


def unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def _sphere_unit(nu, nv):
    th = np.linspace(0, math.pi, nv + 1)
    ph = np.linspace(0, 2 * math.pi, nu, endpoint=False)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    V = np.stack([np.sin(TH) * np.cos(PH), np.cos(TH), np.sin(TH) * np.sin(PH)], -1)
    V = V.reshape(-1, 3).astype(np.float32)
    F = []
    for i in range(nv):
        for j in range(nu):
            j2 = (j + 1) % nu
            F.append((i * nu + j, (i + 1) * nu + j, (i + 1) * nu + j2, i * nu + j2))
    return V, np.array(F, np.int32)


SPH = {"hi": _sphere_unit(22, 14), "mid": _sphere_unit(14, 9), "lo": _sphere_unit(8, 5)}


def ellipsoid(center, R, radii, q="mid"):
    V, F = SPH[q]
    return (V * np.asarray(radii, np.float32)) @ np.asarray(R, np.float32).T + center, F


def sphere(center, r, q="lo"):
    return ellipsoid(center, np.eye(3, dtype=np.float32), (r, r, r), q)


_FR_FACES: dict = {}


def frustum(p0, p1, r0, r1, n=10, cap1=False):
    """Tapered tube from p0 to p1 (open at p0, optional cap at p1)."""
    p0 = np.asarray(p0, np.float32)
    p1 = np.asarray(p1, np.float32)
    a = unit(p1 - p0)
    t = v3(1, 0, 0) if abs(a[0]) < 0.9 else v3(0, 1, 0)
    u = unit(np.cross(a, t))
    w = np.cross(a, u)
    ang = np.linspace(0, 2 * math.pi, n, endpoint=False)
    ring = (np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * w).astype(np.float32)
    V = np.concatenate([p0 + ring * r0, p1 + ring * r1])
    key = (n, cap1)
    if key not in _FR_FACES:
        F = [(i, n + i, n + (i + 1) % n, (i + 1) % n) for i in range(n)]
        if cap1:
            c = 2 * n
            F += [(c, n + (i + 1) % n, n + i, n + i) for i in range(n)]
        _FR_FACES[key] = np.array(F, np.int32)
    F = _FR_FACES[key]
    if cap1:
        V = np.concatenate([V, p1[None]])
    return V, F


def two_bone(S, T, L1, L2, pole):
    d = T - S
    dist = float(np.linalg.norm(d))
    u = d / (dist + 1e-9)
    dist = min(max(dist, abs(L1 - L2) + 0.01), L1 + L2 - 0.003)
    a = (L1 * L1 + dist * dist - L2 * L2) / (2 * dist)
    h = math.sqrt(max(L1 * L1 - a * a, 0.0))
    p = pole - u * float(pole @ u)
    pn = float(np.linalg.norm(p))
    p = p / pn if pn > 1e-4 else v3(0, -1, 0)
    return S + u * a + p * h, S + u * dist


# --------------------------------------------------------------------------- robot model
SIDES = (("LEFT", -1, "left"), ("RIGHT", 1, "right"))
PART_NAMES = ["HEAD", "NECK", "CHEST", "ABDOMEN", "PELVIS"] + [
    f"{side} {k}" for side, _, _ in SIDES
    for k in ("SHOULDER", "UPPER ARM", "ELBOW", "FOREARM", "HAND", "HIP", "THIGH", "KNEE", "SHIN", "FOOT")]
PID = {n: i for i, n in enumerate(PART_NAMES)}
EXTRA = {"HEAD": (0, 0.08, 0), "NECK": (0, 0.07, 0), "CHEST": (0, 0.05, 0), "PELVIS": (0, -0.05, 0)}
PIVOT = v3(0, 1.0, 0)
L_UP, L_FORE = 0.30, 0.27
I3 = np.eye(3, dtype=np.float32)


class RobotModel:
    def __init__(self):
        self.lean = 0.0
        self.head_yaw = 0.0
        self.head_nod = 0.0
        self.head_roll = 0.0
        self.standby = False
        self.idle_pose = True                   # presenting pose (like the photo) when no hands
        self.face_lock = False
        self.arm_pos = {"left": None, "right": None}
        self.arm_goal = {"left": None, "right": None}       # robot-local targets
        self.curl = {"left": [0.2] * 5, "right": [0.2] * 5}
        self.curl_goal = {"left": [0.2] * 5, "right": [0.2] * 5}
        self.arm_raise = {"left": 0.0, "right": 0.0}
        self.leg_thigh = {"left": 0.0, "right": 0.0}
        self.leg_thigh_goal = {"left": 0.0, "right": 0.0}
        self.leg_knee = {"left": 0.0, "right": 0.0}
        self.leg_knee_goal = {"left": 0.0, "right": 0.0}
        self.leg_splay = {"left": 0.0, "right": 0.0}
        self.leg_splay_goal = {"left": 0.0, "right": 0.0}
        self.body_drop = 0.0
        self.body_drop_goal = 0.0
        self.body_pitch = 0.0
        self.body_pitch_goal = 0.0
        self.lean_goal = 0.0
        self.look_yaw = 0.0
        self.look_nod = 0.0
        self.look_roll = 0.0
        self.mouth = 0.0
        self.mouth_goal = 0.0

    def update(self, t):
        for side in ("left", "right"):
            for i in range(5):
                self.curl[side][i] += (self.curl_goal[side][i] - self.curl[side][i]) * 0.45
        sway = 0.0 if self.standby else 0.010 * math.sin(t * 1.3)
        self.lean += (self.lean_goal + sway - self.lean) * 0.12
        self.body_drop += (self.body_drop_goal - self.body_drop) * 0.12
        self.body_pitch += (self.body_pitch_goal - self.body_pitch) * 0.12
        for side in ("left", "right"):
            self.leg_thigh[side] += (self.leg_thigh_goal[side] - self.leg_thigh[side]) * 0.16
            self.leg_knee[side] += (self.leg_knee_goal[side] - self.leg_knee[side]) * 0.16
            self.leg_splay[side] += (self.leg_splay_goal[side] - self.leg_splay[side]) * 0.16
        k = 0.35 if self.face_lock else 0.15
        self.head_yaw += (self.look_yaw - self.head_yaw) * k
        self.head_nod += (self.look_nod - self.head_nod) * k
        self.head_roll += (self.look_roll - self.head_roll) * k
        self.mouth += (self.mouth_goal - self.mouth) * 0.5

    # ---- builds the posed mesh soup; returns V, F, mat, part ids, part (center, radius)
    def build(self, t, explode):
        tmp = [[] for _ in PART_NAMES]
        eye_m = BLACK_M if self.standby else EYE_M

        def emit(name, mesh, mat):
            center = PIVOT - v3(0, self.body_drop, 0)
            turn = rot_x(self.body_pitch)
            vertices = center + (mesh[0] - center) @ turn.T
            tmp[PID[name]].append((vertices, mesh[1], mat))

        M = rot_z(-self.lean)

        def U(p):
            return PIVOT + M @ (np.asarray(p, np.float32) - PIVOT) - v3(0, self.body_drop, 0)

        def ribbed(name, p0, p1, base, amp, n, mat):
            """Bellows: continuous tube whose radius alternates low/high."""
            p0, p1 = np.asarray(p0, np.float32), np.asarray(p1, np.float32)
            for i in range(2 * n):
                a = p0 + (p1 - p0) * (i / (2 * n))
                b = p0 + (p1 - p0) * ((i + 1) / (2 * n))
                ra = base + amp * (i % 2)
                rb_ = base + amp * ((i + 1) % 2)
                emit(name, frustum(a, b, ra, rb_, 12), mat)

        # ---- legs
        for side, sg, key in SIDES:
            thigh_rot = rot_z(sg * self.leg_splay[key]) @ rot_x(self.leg_thigh[key])
            shin_rot = rot_z(sg * self.leg_splay[key]) @ rot_x(
                self.leg_thigh[key] - self.leg_knee[key])
            hip = v3(sg * .098, .905 - self.body_drop, 0)
            knee = hip + thigh_rot @ v3(0, -.405, 0)
            ank = knee + shin_rot @ v3(0, -.395, 0)

            def thigh_point(p):
                return hip + thigh_rot @ np.asarray(p, np.float32)

            def shin_point(p):
                return knee + shin_rot @ np.asarray(p, np.float32)

            emit(f"{side} HIP", sphere(hip, .072, "mid"), BLACK_M)
            emit(f"{side} HIP", ellipsoid(hip + thigh_rot @ v3(sg * .02, -.06, 0), thigh_rot,
                                          (.06, .04, .06), "lo"), BLACK_M)
            emit(f"{side} THIGH", frustum(thigh_point(v3(0, -.045, 0)), knee + thigh_rot @ v3(0, .070, 0),
                                          .090, .070, 16), PANTS_M)
            emit(f"{side} THIGH", frustum(hip, knee, .072, .056, 14), PANTS_M)
            emit(f"{side} THIGH", ellipsoid(thigh_point(v3(-sg * .046, -.175, -.012)), thigh_rot,
                                            (.022, .105, .05), "lo"), BLACK_M)
            emit(f"{side} THIGH", frustum(thigh_point(v3(-sg * .046, -.065, -.071)),
                                          thigh_point(v3(-sg * .046, -.315, -.058)),
                                          .003, .003, 6), FABRIC_TRIM_M)
            emit(f"{side} THIGH", ellipsoid(thigh_point(v3(-sg * .073, -.235, -.060)), thigh_rot,
                                            (.035, .006, .003), "lo"),
                 FABRIC_TRIM_M)
            emit(f"{side} KNEE", sphere(knee, .058, "mid"), BLACK_M)
            emit(f"{side} KNEE", ellipsoid(knee + thigh_rot @ v3(sg * .05, 0, 0), thigh_rot,
                                            (.016, .047, .047), "lo"), BLACK_M)
            emit(f"{side} KNEE", ellipsoid(knee + thigh_rot @ v3(sg * .062, 0, 0), thigh_rot,
                                            (.008, .028, .028), "lo"), METAL_M)
            emit(f"{side} KNEE", ellipsoid(knee + thigh_rot @ v3(0, .0, -.044), thigh_rot,
                                            (.04, .05, .026), "lo"), BLACK_M)
            emit(f"{side} SHIN", frustum(shin_point(v3(0, -.070, 0)), shin_point(v3(0, .035, 0)),
                                         .073, .052, 16), PANTS_M)
            emit(f"{side} SHIN", frustum(knee, ank, .058, .04, 14), PANTS_M)
            emit(f"{side} SHIN", ellipsoid(shin_point(v3(-sg * .030, -.115, -.022)), shin_rot,
                                           (.022, .085, .04), "lo"), BLACK_M)
            emit(f"{side} SHIN", frustum(shin_point(v3(-sg * .030, -.09, -.059)),
                                         shin_point(v3(-sg * .030, -.32, -.043)),
                                         .003, .003, 6), FABRIC_TRIM_M)
            emit(f"{side} SHIN", ellipsoid(shin_point(v3(sg * .005, -.33, 0)), shin_rot,
                                           (.046, .012, .046), "lo"),
                 FABRIC_TRIM_M)
            emit(f"{side} FOOT", sphere(ank, .043, "mid"), BLACK_M)
            emit(f"{side} FOOT", ellipsoid(ank + shin_rot @ v3(sg * .005, -.055, -.05), shin_rot,
                                           (.05, .042, .11), "mid"), WHITE_M)
            emit(f"{side} FOOT", ellipsoid(ank + shin_rot @ v3(sg * .005, -.083, -.055), shin_rot,
                                           (.053, .014, .118), "lo"), METAL_M)
        emit("PELVIS", ellipsoid(U(v3(0, .965, 0)), M, (.165, .068, .105), "hi"), WHITE_M)
        emit("PELVIS", ellipsoid(U(v3(0, .925, -.03)), M, (.12, .06, .08), "mid"), WHITE_M)

        # ---- trunk and tailored shirt overlays
        ribbed("ABDOMEN", U(v3(0, .995, 0)), U(v3(0, 1.17, 0)), .078, .014, 4, BLACK_M)
        emit("CHEST", ellipsoid(U(v3(0, 1.305, 0)), M, (.225, .190, .158), "hi"), SHIRT_M)
        emit("CHEST", ellipsoid(U(v3(0, 1.355, 0)), M, (.200, .125, .122), "hi"), SHIRT_M)
        emit("CHEST", ellipsoid(U(v3(0, 1.255, -.012)), M, (.178, .082, .112), "hi"), SHIRT_M)
        emit("CHEST", ellipsoid(U(v3(0, 1.175, -.01)), M, (.150, .062, .100), "mid"), SHIRT_M)
        for y in (1.34, 1.29, 1.24):
            emit("CHEST", ellipsoid(U(v3(0, y, -.160)), M, (.006, .006, .003), "lo"), FABRIC_TRIM_M)
        emit("CHEST", frustum(U(v3(0, 1.405, -.105)), U(v3(0, 1.455, -.092)),
                              .045, .040, 12), FABRIC_TRIM_M)
        emit("CHEST", ellipsoid(U(v3(0, 1.145, 0)), M, (.190, .014, .123), "mid"), FABRIC_TRIM_M)
        for x in (-.11, .11):
            emit("CHEST", ellipsoid(U(v3(x, 1.375, -.105)), M, (.004, .035, .003), "lo"),
                 FABRIC_TRIM_M)
        ribbed("NECK", U(v3(0, 1.455, 0)), U(v3(0, 1.57, 0)), .036, .005, 3, BLACK_M)

        # ---- head (human-like face, ear pods) - follows YOUR head
        Rh = M @ rot_z(self.head_roll) @ rot_y(self.head_yaw) @ rot_x(self.head_nod)
        hc = U(v3(0, 1.665, 0))

        def H(x, y, z):
            return hc + Rh @ v3(x, y, z)

        emit("HEAD", ellipsoid(hc, Rh, (.082, .108, .093), "hi"), WHITE_M)
        emit("HEAD", ellipsoid(H(0, -.05, -.018), Rh, (.064, .062, .072), "hi"), WHITE_M)   # jaw / chin
        emit("HEAD", ellipsoid(H(0, -.006, -.088), Rh, (.011, .022, .014), "lo"), WHITE_M)  # nose
        emit("HEAD", ellipsoid(H(0, -.052, -.082), Rh, (.019, .004 + .014 * self.mouth, .006), "lo"),
             BLACK_M if self.mouth > .12 else METAL_M)                                       # mouth
        emit("HEAD", ellipsoid(H(0, .052, .006), Rh, (.085, .065, .096), "hi"), HAIR_M)
        emit("HEAD", ellipsoid(H(0, .068, -.078), Rh, (.055, .018, .018), "mid"), HAIR_M)
        for sx in (-1, 1):
            emit("HEAD", ellipsoid(H(sx * .052, .078, -.018), Rh, (.003, .035, .003), "lo"),
                 FABRIC_TRIM_M)
        for sx in (-1, 1):
            emit("HEAD", ellipsoid(H(sx * .031, .018, -.085), Rh, (.017, .007, .006), "lo"), METAL_M)  # sockets
            emit("HEAD", ellipsoid(H(sx * .031, .018, -.089), Rh, (.008, .005, .004), "lo"), eye_m)    # eyes
            emit("HEAD", ellipsoid(H(sx * .086, .0, .01), Rh, (.016, .042, .042), "mid"), BLACK_M)     # ear pod
            emit("HEAD", ellipsoid(H(sx * .098, .0, .01), Rh, (.006, .028, .028), "lo"), METAL_M)      # pod cover

        # ---- arms (3D inverse kinematics to the tracked hands)
        for side, sg, key in SIDES:
            S = U(v3(sg * .215, 1.395, 0))
            emit(f"{side} SHOULDER", sphere(S, .058, "mid"), BLACK_M)
            emit(f"{side} SHOULDER", ellipsoid(S + M @ v3(sg * .03, .022, 0), M, (.05, .064, .064), "mid"), WHITE_M)
            rest = S + v3(sg * .07, -.55, -.03)
            if sg < 0 and self.idle_pose and not self.standby:
                rest = S + v3(-.30, -.03, -.20)            # presenting pose from the photo
            goal = self.arm_goal[key] if (self.arm_goal[key] is not None and not self.standby) else rest
            if self.arm_pos[key] is None:
                self.arm_pos[key] = rest.copy()
            self.arm_pos[key] = self.arm_pos[key] + (goal - self.arm_pos[key]) * 0.35
            E, Wr = two_bone(S, self.arm_pos[key], L_UP, L_FORE, v3(sg * .5, -1, .6))
            self.arm_raise[key] = math.degrees(math.acos(float(np.clip(unit(E - S) @ v3(0, -1, 0), -1, 1))))
            emit(f"{side} UPPER ARM", frustum(S, E, .05, .041, 14), WHITE_M)
            emit(f"{side} ELBOW", sphere(E, .043, "mid"), BLACK_M)
            emit(f"{side} ELBOW", ellipsoid(E + v3(sg * .035, 0, 0), I3, (.01, .03, .03), "lo"), METAL_M)
            Wc = E + (Wr - E) * 0.82
            emit(f"{side} UPPER ARM", frustum(S + (E - S) * .04, S + (E - S) * .52,
                                               .060, .051, 14), SHIRT_M)
            emit(f"{side} FOREARM", frustum(E, Wc, .041, .033, 14), WHITE_M)
            emit(f"{side} FOREARM", frustum(Wc, Wr, .032, .03, 12), BLACK_M)
            self._hand(emit, f"{side} HAND", Wr, unit(Wr - E), sg, self.curl[key])

        # ---- per-part centre, explode offset, flatten
        Vs, Fs, Ms, Ps, base = [], [], [], [], 0
        info = []
        for pid, items in enumerate(tmp):
            allv = np.concatenate([it[0] for it in items])
            lo, hi = allv.min(0), allv.max(0)
            c = (lo + hi) / 2
            r = float(np.linalg.norm(hi - lo)) * 0.42
            off = np.zeros(3, np.float32)
            if explode > 1e-3:
                v = c - PIVOT
                off = v3(v[0] * 1.5, v[1] * 0.6, v[2] * 1.2) + v3(*EXTRA.get(PART_NAMES[pid], (0, 0, 0)))
                off = off * explode + v3(0, 0.5 * explode, 0)
            info.append((c + off, max(r, 0.04)))
            for V, F, mat in items:
                Vs.append(V + off)
                Fs.append(F + base)
                Ms.append(np.full(len(F), mat, np.int8))
                Ps.append(np.full(len(F), pid, np.int16))
                base += len(V)
        return (np.concatenate(Vs), np.concatenate(Fs), np.concatenate(Ms),
                np.concatenate(Ps), info)

    @staticmethod
    def _hand(emit, name, Wr, f, sg, curls):
        n = v3(-sg, 0, 0)
        n = n - f * float(f @ n)
        if np.linalg.norm(n) < 0.3:
            n0 = v3(0, 0, -1)
            n = n0 - f * float(f @ n0)
        n = unit(n)
        l = unit(np.cross(f, n))
        if l[2] > 0:
            l = -l
        R = np.stack([l, f, np.cross(l, f)], 1).astype(np.float32)
        emit(name, ellipsoid(Wr + f * .045, R, (.034, .046, .016), "lo"), WHITE_M)
        lens = (.038, .026, .020)
        for i in range(4):
            lat = (.028, .010, -.009, -.027)[i]
            scale = (1.0, 1.08, 1.0, 0.8)[i]
            c = float(curls[i + 1])
            p = Wr + f * .085 + l * lat + n * .006
            acc = 0.0
            for k in range(3):
                emit(name, sphere(p, .0085), BLACK_M)                 # knuckle joint
                acc += c * (0.85, 1.05, 0.8)[k]
                d = f * math.cos(acc) + n * math.sin(acc)
                q = p + d * lens[k] * scale
                emit(name, frustum(p, q, .0092 - .0012 * k, .0082 - .0012 * k, 6, cap1=(k == 2)), WHITE_M)
                p = q
        c = float(curls[0])
        p = Wr + f * .04 + l * .042
        d0 = unit(f * .6 + l * .7 + n * .15)
        for k, ln in enumerate((.034, .027)):
            emit(name, sphere(p, .0095), BLACK_M)
            a = c * (0.9 if k == 0 else 1.4)
            d = unit(d0 * math.cos(a) + n * math.sin(a))
            q = p + d * ln
            emit(name, frustum(p, q, .011 - .0015 * k, .0095 - .0015 * k, 6, cap1=(k == 1)), WHITE_M)
            p = q


# --------------------------------------------------------------------------- camera / renderer
class View:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.cx, self.cy = w / 2.0, h / 2.0 + 10
        self.f = 1.25 * h
        self.D0 = 2.7
        self.yaw, self.pitch = 0.0, 0.05
        self.zoom, self.zoom_goal = 1.0, 1.0
        self.T = v3(0, .9, 0)
        self.T_goal = v3(0, .9, 0)
        self.auto = False
        self.fit = 1.0
        self.Rm = np.eye(3, dtype=np.float32)
        self.D = self.D0

    def update(self, dt, explode):
        if self.auto:
            self.yaw += dt * 0.6
        self.zoom_goal = float(np.clip(self.zoom_goal, 0.6, 7.0))
        self.zoom += (self.zoom_goal - self.zoom) * 0.18
        self.T += (self.T_goal - self.T) * 0.18
        self.fit = 1.0 / (1.0 + 0.5 * explode)
        self.Rm = rot_x(self.pitch) @ rot_y(self.yaw)
        self.D = max(self.D0 / (self.zoom * self.fit), 0.4)

    def to_view(self, P):
        Pv = (np.asarray(P, np.float32) - self.T) @ self.Rm.T
        Pv[..., 2] += self.D
        return Pv

    def project(self, P):
        Pv = self.to_view(P)
        Z = np.maximum(Pv[..., 2], 0.06)
        return self.cx + self.f * Pv[..., 0] / Z, self.cy - self.f * Pv[..., 1] / Z, Pv[..., 2]

    def unproject_local(self, px, py):
        """Screen point -> robot-local point on the plane through the camera target."""
        Xv = (px - self.cx) * self.D / self.f
        Yv = -(py - self.cy) * self.D / self.f
        return self.T + self.Rm.T @ v3(Xv, Yv, 0)

    LIGHT = unit(v3(-0.35, 0.55, -0.75))
    FILL = unit(v3(0.6, -0.2, -0.5))

    def draw_soup(self, L, E, V, F, mat, pid, sel, hov, dim):
        """Glass-hologram renderer: shaded ice-teal fill into L, glowing wireframe into E
        (back edges are drawn faintly so the body looks like see-through glass)."""
        Pv = self.to_view(V)
        Z = np.maximum(Pv[:, 2], 0.06)
        xs = self.cx + self.f * Pv[:, 0] / Z
        ys = self.cy - self.f * Pv[:, 1] / Z
        ok = (Pv[:, 2] > 0.08)[F].all(1)
        Q = Pv[F]
        c = Q.mean(1)
        nrm = np.cross(Q[:, 2] - Q[:, 0], Q[:, 3] - Q[:, 1])
        ln = np.linalg.norm(nrm, axis=1)
        facing = -(nrm * c).sum(1)
        good = ok & (ln > 1e-9)
        vis = np.nonzero(good & (facing > 0))[0]
        back = np.nonzero(good & (facing <= 0))[0]
        P2 = np.clip(np.stack([xs[F], ys[F]], -1), -20000, 20000).astype(np.int32)     # (nF, 4, 2)
        if len(back):
            bcol = tuple(int(x * dim) for x in (66, 76, 24))
            cv2.polylines(E, P2[back], True, bcol, 1, cv2.LINE_8)
        if len(vis) == 0:
            return
        nn = nrm[vis] / ln[vis, None]
        cc = c[vis]
        vd = -cc / (np.linalg.norm(cc, axis=1, keepdims=True) + 1e-9)
        diff = np.clip(nn @ self.LIGHT, 0, 1)
        fill = np.clip(nn @ self.FILL, 0, 1)
        ndv = np.clip((nn * vd).sum(1), 0, 1)
        hv = self.LIGHT[None] + vd
        hv /= np.linalg.norm(hv, axis=1, keepdims=True) + 1e-9
        spec = np.clip((nn * hv).sum(1), 0, 1) ** 28
        m = mat[vis]
        tone = 0.34 + 0.60 * diff + 0.16 * fill
        col = (HOLO_COL[m] * tone[:, None]
               + (MAT_SPEC[m] * spec * 170)[:, None]
               + ((1 - ndv) ** 2.2)[:, None] * np.array(CYAN, np.float32) * 0.75)
        col[m == EYE_M] = HOLO_COL[EYE_M]
        p = pid[vis]
        if sel >= 0:
            k = p == sel
            col[k] = col[k] * 0.70 + np.array(ORANGE, np.float32) * 0.30
        if hov >= 0 and hov != sel:
            k = p == hov
            col[k] = col[k] * 0.72 + np.array(CYAN, np.float32) * 0.28
        col *= dim
        order = np.argsort(-cc[:, 2])
        pts = P2[vis][order]
        colors = np.clip(col[order], 0, 255).astype(np.uint8).tolist()
        fillp = cv2.fillConvexPoly
        for poly, color in zip(pts, colors):
            fillp(L, poly, color)
        # glowing wireframe of the visible faces
        pvis = P2[vis]
        normal = (p != sel) & (p != hov if hov != sel else True)
        if sel >= 0 and hov == sel:
            normal = p != sel
        if normal.any():
            cv2.polylines(E, pvis[normal], True, tuple(int(x * dim) for x in (234, 242, 96)), 1, AA)
        if sel >= 0 and (p == sel).any():
            cv2.polylines(E, pvis[p == sel], True, tuple(int(x * dim) for x in (70, 175, 255)), 1, AA)
        if hov >= 0 and hov != sel and (p == hov).any():
            cv2.polylines(E, pvis[p == hov], True, (255, 255, 170), 1, AA)


# --------------------------------------------------------------------------- hand + face input
def _d(a, b):
    return float(math.hypot(a.x - b.x, a.y - b.y))


def finger_curls(lm):
    out = [float(np.clip((1.8 - _d(lm[4], lm[17]) / max(_d(lm[2], lm[17]), 1e-6)) / 0.8, 0, 1))]
    for tip, mcp in ((8, 5), (12, 9), (16, 13), (20, 17)):
        ratio = _d(lm[tip], lm[0]) / max(_d(lm[mcp], lm[0]), 1e-6)
        out.append(float(np.clip((1.85 - ratio) / 0.95, 0, 1)))
    return out


def count_fingers(lm):
    n = 1 if _d(lm[4], lm[17]) > _d(lm[3], lm[17]) else 0
    for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
        if _d(lm[tip], lm[0]) > _d(lm[pip], lm[0]) * 1.05:
            n += 1
    return n


def parse_hands(result, w, h):
    found = []
    for pts in (result.multi_hand_landmarks or []):
        lm = pts.landmark
        found.append({
            "lm": [(p.x, p.y) for p in lm],
            "wrist": np.array([lm[0].x * w, lm[0].y * h], np.float32),
            "tip": np.array([lm[8].x * w, lm[8].y * h], np.float32),
            "pointer": np.array([(lm[4].x + lm[8].x) / 2 * w, (lm[4].y + lm[8].y) / 2 * h], np.float32),
            "curls": finger_curls(lm),
            "count": count_fingers(lm),
            "pinch": _d(lm[4], lm[8]) / max(_d(lm[0], lm[9]), 1e-6) < 0.38,
        })
    found.sort(key=lambda hd: hd["wrist"][0])
    if not found:
        return {}
    if len(found) == 1:
        return {("left" if found[0]["wrist"][0] < w / 2 else "right"): found[0]}
    return {"left": found[0], "right": found[-1]}


def parse_face(result):
    """Raw head pose ratios from MediaPipe FaceMesh (frame is already mirrored)."""
    if not result or not result.multi_face_landmarks:
        return None
    lm = result.multi_face_landmarks[0].landmark
    xl, xr = sorted((lm[234].x, lm[454].x))
    fw = max(xr - xl, 1e-4)
    mid = (xl + xr) / 2
    top, chin = lm[10], lm[152]
    fh = max(chin.y - top.y, 1e-4)
    a, b = lm[33], lm[263]
    if a.x > b.x:
        a, b = b, a
    return {
        "yaw_r": (lm[1].x - mid) / fw,
        "pitch_r": (lm[1].y - (top.y + chin.y) / 2) / fh,
        "roll": math.atan2(b.y - a.y, b.x - a.x),
        "center": (mid, (top.y + chin.y) / 2),
        "box": (xl, top.y, xr, chin.y),
        "pts": [(lm[i].x, lm[i].y) for i in range(0, 468, 5)],
    }


# --------------------------------------------------------------------------- 2D drawing helpers
def put(image, text, org, scale=0.55, color=WHITE, thickness=1):
    if not text.isascii():
        put_unicode(image, text, org, scale, color, thickness)
        return
    cv2.putText(image, text, org, FONT, scale, (0, 0, 0), thickness + 2, AA)
    cv2.putText(image, text, org, FONT, scale, color, thickness, AA)


def _script_for_text(text: str) -> str:
    for char in text:
        code = ord(char)
        if 0x0D00 <= code <= 0x0D7F:
            return "malayalam"
        if 0x0B80 <= code <= 0x0BFF:
            return "tamil"
        if 0x0900 <= code <= 0x097F:
            return "devanagari"
        if 0x0980 <= code <= 0x09FF:
            return "bengali"
        if 0x0A00 <= code <= 0x0A7F:
            return "gurmukhi"
        if 0x0A80 <= code <= 0x0AFF:
            return "gujarati"
        if 0x0C00 <= code <= 0x0C7F:
            return "telugu"
        if 0x0C80 <= code <= 0x0CFF:
            return "kannada"
        if 0x0600 <= code <= 0x08FF:
            return "arabic"
        if 0x0590 <= code <= 0x05FF:
            return "hebrew"
        if 0x0E00 <= code <= 0x0E7F:
            return "thai"
        if 0x3040 <= code <= 0x30FF:
            return "japanese"
        if 0x4E00 <= code <= 0x9FFF:
            return "chinese"
        if 0xAC00 <= code <= 0xD7AF:
            return "korean"
        if 0x0400 <= code <= 0x052F:
            return "cyrillic"
    return "default"


_UNICODE_FONT_PATHS = {
    "malayalam": ("/System/Library/Fonts/Supplemental/Malayalam Sangam MN.ttc",
                  "/System/Library/Fonts/Supplemental/Malayalam MN.ttc"),
    "tamil": ("/System/Library/Fonts/Supplemental/Tamil Sangam MN.ttc",
              "/System/Library/Fonts/Supplemental/Tamil MN.ttc"),
    "devanagari": ("/System/Library/Fonts/Supplemental/Devanagari Sangam MN.ttc",
                   "/System/Library/Fonts/Supplemental/DevanagariMT.ttc"),
    "bengali": ("/System/Library/Fonts/Supplemental/Bangla Sangam MN.ttc",),
    "gurmukhi": ("/System/Library/Fonts/Supplemental/Gurmukhi Sangam MN.ttc",),
    "gujarati": ("/System/Library/Fonts/Supplemental/Gujarati Sangam MN.ttc",),
    "telugu": ("/System/Library/Fonts/Supplemental/Telugu Sangam MN.ttc",),
    "kannada": ("/System/Library/Fonts/Supplemental/Kannada Sangam MN.ttc",),
    "arabic": ("/System/Library/Fonts/SFArabic.ttf",),
    "hebrew": ("/System/Library/Fonts/SFHebrew.ttf",),
    "japanese": ("/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",),
    "chinese": ("/System/Library/Fonts/PingFang.ttc",),
    "korean": ("/System/Library/Fonts/AppleSDGothicNeo.ttc",),
    "cyrillic": ("/System/Library/Fonts/Supplemental/Arial.ttf",),
    "default": (),
}
_UNICODE_FALLBACK_FONTS = (
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "C:/Windows/Fonts/arial.ttf",
)
_NATIVE_FONT_FAMILIES = {
    "malayalam": "Malayalam Sangam MN",
    "tamil": "Tamil Sangam MN",
    "devanagari": "Devanagari Sangam MN",
    "bengali": "Bangla Sangam MN",
    "gurmukhi": "Gurmukhi Sangam MN",
    "gujarati": "Gujarati Sangam MN",
    "telugu": "Telugu Sangam MN",
    "kannada": "Kannada Sangam MN",
    "arabic": "Geeza Pro",
    "hebrew": "SF Hebrew",
    "japanese": "Hiragino Sans",
    "chinese": "PingFang SC",
    "korean": "Apple SD Gothic Neo",
    "cyrillic": "Arial",
}


@lru_cache(maxsize=32)
def _native_unicode_font(script: str, size: int):
    family = _NATIVE_FONT_FAMILIES.get(script)
    font = NSFont.fontWithName_size_(family, size) if family else None
    return font or NSFont.systemFontOfSize_(size)


@lru_cache(maxsize=128)
def _unicode_font(script: str, size: int):
    candidates = _UNICODE_FONT_PATHS.get(script, ()) + _UNICODE_FALLBACK_FONTS
    for path in candidates:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default(size=size)


def _text_width(text: str, scale: float) -> float:
    if text.isascii():
        return float(cv2.getTextSize(text, FONT, scale, 1)[0][0])
    size = max(10, round(scale * 30))
    if NSBitmapImageRep is not None:
        font = _native_unicode_font(_script_for_text(text), size)
        return float(NSString.stringWithString_(text).sizeWithAttributes_(
            {NSFontAttributeName: font}).width)
    font = _unicode_font(_script_for_text(text), size)
    return float(font.getlength(text))


def wrap_display_text(text: str, max_width: float, scale: float) -> list[str]:
    lines = []
    line = ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if line and _text_width(candidate, scale) > max_width:
            lines.append(line)
            line = ""
        for char in word:
            candidate = line + char
            if line and _text_width(candidate, scale) > max_width:
                lines.append(line)
                line = char
            else:
                line = candidate
        if line and word != text.split()[-1]:
            line += " "
    if line.strip() or not lines:
        lines.append(line.strip())
    return lines


def fit_display_text(text: str, max_width: float, scale: float) -> str:
    if _text_width(text, scale) <= max_width:
        return text
    suffix = "..."
    fitted = text.rstrip()
    while fitted and _text_width(fitted + suffix, scale) > max_width:
        fitted = fitted[:-1].rstrip()
    return fitted + suffix if fitted else suffix


def put_unicode(image, text, org, scale=0.55, color=WHITE, thickness=1):
    x, baseline = map(int, org)
    size = max(10, round(scale * 30))
    if NSBitmapImageRep is not None:
        _put_native_unicode(image, text, x, baseline, size, color, thickness)
        return
    font = _unicode_font(_script_for_text(text), size)
    bounds = font.getbbox(text, anchor="ls")
    pad = max(thickness + 2, 2)
    left = max(0, x + bounds[0] - pad)
    top = max(0, baseline + bounds[1] - pad)
    right = min(image.shape[1], x + bounds[2] + pad + 1)
    bottom = min(image.shape[0], baseline + bounds[3] + pad + 1)
    if left >= right or top >= bottom:
        return
    rgb = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2RGB)
    patch = Image.fromarray(rgb)
    draw = ImageDraw.Draw(patch)
    origin = (x - left, baseline - top)
    draw.text(origin, text, font=font, anchor="ls", fill=(0, 0, 0),
              stroke_width=thickness + 2, stroke_fill=(0, 0, 0))
    draw.text(origin, text, font=font, anchor="ls", fill=tuple(color[::-1]),
              stroke_width=max(thickness - 1, 0), stroke_fill=tuple(color[::-1]))
    image[top:bottom, left:right] = cv2.cvtColor(np.asarray(patch), cv2.COLOR_RGB2BGR)


def _put_native_unicode(image, text, x, baseline, size, color, thickness):
    script = _script_for_text(text)
    font = _native_unicode_font(script, size)
    attributes = {NSFontAttributeName: font}
    string = NSString.stringWithString_(text)
    width = max(1, int(math.ceil(string.sizeWithAttributes_(attributes).width)))
    pad = max(thickness + 2, 2)
    ascent = int(math.ceil(font.ascender()))
    descent = int(math.ceil(-font.descender()))
    left = max(0, x - pad)
    top = max(0, baseline - ascent - pad)
    right = min(image.shape[1], x + width + pad + 1)
    bottom = min(image.shape[0], baseline + descent + pad + 1)
    if left >= right or top >= bottom:
        return
    patch_width, patch_height = right - left, bottom - top
    bitmap = NSBitmapImageRep.alloc().initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel_(
        None, patch_width, patch_height, 8, 4, True, False, NSDeviceRGBColorSpace, 0, 0)
    context = NSGraphicsContext.graphicsContextWithBitmapImageRep_(bitmap)
    shadow_attributes = {
        NSFontAttributeName: font,
        NSForegroundColorAttributeName: NSColor.blackColor(),
    }
    attributes[NSForegroundColorAttributeName] = NSColor.colorWithCalibratedRed_green_blue_alpha_(
        color[2] / 255.0, color[1] / 255.0, color[0] / 255.0, 1.0)
    NSGraphicsContext.saveGraphicsState()
    try:
        NSGraphicsContext.setCurrentContext_(context)
        text_origin = NSMakePoint(x - left, patch_height - (baseline - top))
        NSAttributedString.alloc().initWithString_attributes_(text, shadow_attributes).drawAtPoint_(
            NSMakePoint(text_origin.x + 1, text_origin.y - 1))
        NSAttributedString.alloc().initWithString_attributes_(text, attributes).drawAtPoint_(text_origin)
    finally:
        NSGraphicsContext.restoreGraphicsState()
    raw = np.frombuffer(bitmap.bitmapData(), dtype=np.uint8)
    rgba = raw.reshape(patch_height, bitmap.bytesPerRow())[:, :patch_width * 4]
    rgba = rgba.reshape(patch_height, patch_width, 4)
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    base = image[top:bottom, left:right].astype(np.float32)
    overlay = rgba[:, :, :3][:, :, ::-1].astype(np.float32)
    image[top:bottom, left:right] = np.clip(
        overlay + base * (1.0 - alpha), 0, 255).astype(np.uint8)


def brackets(image, x0, y0, x1, y1, color=CYAN, ln=14, th=2):
    for cx, cy, sx, sy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        cv2.line(image, (cx, cy), (cx + ln * sx, cy), color, th, AA)
        cv2.line(image, (cx, cy), (cx, cy + ln * sy), color, th, AA)


# --------------------------------------------------------------------------- premium UI kit
_UI = {"frost": None}        # per-frame blurred copy of the scene = frosted-glass backdrop


def txt(image, text, org, scale=0.4, color=INK1, th=1, font=FONT, shadow=True):
    """Clean anti-aliased UI text with an optional 1px soft shadow."""
    if not text.isascii():
        put_unicode(image, text, org, scale, color, th)
        return
    x, y = int(org[0]), int(org[1])
    if shadow:
        cv2.putText(image, text, (x, y + 1), font, scale, (4, 6, 3), th, AA)
    cv2.putText(image, text, (x, y), font, scale, color, th, AA)


def tw(text, scale=0.4, font=FONT, th=1) -> int:
    return int(cv2.getTextSize(text, font, scale, th)[0][0])


def tracked_width(text, scale=0.3, track=2) -> int:
    if not text:
        return 0
    return sum(tw(ch, scale) + track for ch in text) - track


def put_tracked(image, text, org, scale=0.3, color=INK3, track=2, th=1) -> int:
    """Letter-spaced small caps label. Returns the x where the text ended."""
    x, y = int(org[0]), int(org[1])
    for ch in text:
        cv2.putText(image, ch, (x, y), FONT, scale, color, th, AA)
        x += tw(ch, scale) + track
    return x - track


@lru_cache(maxsize=96)
def _rmask(pw, ph, r):
    m = np.zeros((ph, pw), np.uint8)
    r = int(max(0, min(r, pw // 2, ph // 2)))
    cv2.rectangle(m, (r, 0), (pw - r - 1, ph - 1), 255, -1)
    cv2.rectangle(m, (0, r), (pw - 1, ph - r - 1), 255, -1)
    for cx in (r, pw - r - 1):
        for cy in (r, ph - r - 1):
            cv2.circle(m, (cx, cy), r, 255, -1, AA)
    return m


@lru_cache(maxsize=48)
def _cmask(pw, ph, c):
    """Sci-fi panel mask: top-left and bottom-right corners are chamfered."""
    m = np.zeros((ph, pw), np.uint8)
    c = int(max(2, min(c, pw // 3, ph // 3)))
    pts = np.array([[c, 0], [pw - 1, 0], [pw - 1, ph - 1 - c], [pw - 1 - c, ph - 1],
                    [0, ph - 1], [0, c]], np.int32)
    cv2.fillPoly(m, [pts], 255, AA)
    return m


def tint_rrect(image, x0, y0, x1, y1, r, color, alpha=0.5):
    """Translucent rounded rectangle."""
    H, W = image.shape[:2]
    x0, y0, x1, y1 = map(int, (x0, y0, x1, y1))
    pw, ph = x1 - x0, y1 - y0
    if pw < 2 or ph < 2:
        return
    m = _rmask(pw, ph, int(r))
    cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    mm = m[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0].astype(np.float32)[..., None] / 255.0 * alpha
    roi = image[cy0:cy1, cx0:cx1]
    roi[:] = (roi * (1 - mm) + np.array(color, np.float32) * mm).astype(np.uint8)


def rrect_line(image, x0, y0, x1, y1, r, color, th=1):
    """Anti-aliased rounded-rectangle outline."""
    x0, y0, x1, y1 = map(int, (x0, y0, x1, y1))
    r = int(max(0, min(r, (x1 - x0) // 2, (y1 - y0) // 2)))
    cv2.line(image, (x0 + r, y0), (x1 - r, y0), color, th, AA)
    cv2.line(image, (x0 + r, y1), (x1 - r, y1), color, th, AA)
    cv2.line(image, (x0, y0 + r), (x0, y1 - r), color, th, AA)
    cv2.line(image, (x1, y0 + r), (x1, y1 - r), color, th, AA)
    if r > 0:
        for cx, cy, a0 in ((x0 + r, y0 + r, 180), (x1 - r, y0 + r, 270),
                           (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90)):
            cv2.ellipse(image, (cx, cy), (r, r), 0, a0, a0 + 90, color, th, AA)


@lru_cache(maxsize=32)
def _glass_assets(pw, ph, r):
    pad = 16
    mask = _cmask(pw, ph, r)
    big = np.zeros((ph + 2 * pad, pw + 2 * pad), np.uint8)
    big[pad + 5:pad + 5 + ph, pad:pad + pw] = mask
    shadow = cv2.GaussianBlur(big, (0, 0), 8).astype(np.float32) / 255.0
    edge = cv2.subtract(mask, cv2.erode(mask, np.ones((3, 3), np.uint8))) > 0
    t = np.linspace(0, 1, ph, dtype=np.float32)[:, None, None]
    grad = np.array((40, 32, 25), np.float32) * (1 - t) + np.array((27, 21, 16), np.float32) * t
    grad = np.broadcast_to(grad, (ph, pw, 3)).copy()
    ecol = np.array((125, 106, 85), np.float32) * (1 - t) + np.array((78, 70, 57), np.float32) * t
    ecol = np.broadcast_to(ecol, (ph, pw, 3)).copy()
    return shadow, mask.astype(np.float32)[..., None] / 255.0, edge, grad, ecol


def glass_panel(image, x0, y0, x1, y1, title=None, accent=CYAN, radius=14, alpha=0.88,
                rule_end=None):
    """Draw a clean glass card with a soft shadow, restrained border, and letter-spaced title."""
    H, W = image.shape[:2]
    x0, y0 = max(int(x0), 0), max(int(y0), 0)
    x1, y1 = min(int(x1), W), min(int(y1), H)
    pw, ph = x1 - x0, y1 - y0
    if pw < 24 or ph < 24:
        return
    r = int(min(radius, pw // 3, ph // 3))
    shadow, mask, edge, grad, ecol = _glass_assets(pw, ph, r)
    pad = 16
    sx0, sy0 = max(x0 - pad, 0), max(y0 - pad, 0)
    sx1, sy1 = min(x1 + pad, W), min(y1 + pad, H)
    sh = shadow[sy0 - (y0 - pad):sy1 - (y0 - pad), sx0 - (x0 - pad):sx1 - (x0 - pad)]
    reg = image[sy0:sy1, sx0:sx1]
    reg[:] = (reg * (1 - 0.55 * sh[..., None])).astype(np.uint8)
    roi = image[y0:y1, x0:x1]
    frost = _UI["frost"]
    if frost is not None and frost.shape[:2] == (H, W):
        back = frost[y0:y1, x0:x1].astype(np.float32)
    else:
        back = roi.astype(np.float32)
    body = back * 0.34 + grad * 0.66
    a = mask * alpha
    out = roi.astype(np.float32) * (1 - a) + body * a
    out[edge] = out[edge] * 0.42 + ecol[edge] * 0.58
    roi[:] = np.clip(out, 0, 255).astype(np.uint8)
    if accent is not None:
        cv2.line(image, (x0 + r + 3, y0), (x0 + r + 42, y0), accent, 1, AA)
        ln = 8
        cv2.line(image, (x1 - 1, y0), (x1 - 1 - ln, y0), accent, 1, AA)
        cv2.line(image, (x1 - 1, y0), (x1 - 1, y0 + ln), accent, 1, AA)
        cv2.line(image, (x0, y1 - 1), (x0 + ln, y1 - 1), accent, 1, AA)
        cv2.line(image, (x0, y1 - 1), (x0, y1 - 1 - ln), accent, 1, AA)
    if title:
        cv2.circle(image, (x0 + 19, y0 + 20), 3, accent if accent is not None else CYAN, -1, AA)
        rx = rule_end if rule_end is not None else x1 - 16
        label = title.upper()
        maxw = rx - (x0 + 30) - 6
        while len(label) > 3 and tracked_width(label, 0.32, 2) > maxw:
            label = label[:-1]
        ex = put_tracked(image, label, (x0 + 30, y0 + 24), 0.32, INK2, 2)
        if ex + 10 < rx:
            cv2.line(image, (ex + 10, y0 + 20), (rx, y0 + 20), HAIR, 1, AA)


def hud_panel(image, x0, y0, x1, y1, title=None):        # kept for compatibility
    glass_panel(image, x0, y0, x1, y1, title)


def pill_width(label, scale=0.34) -> int:
    return tw(label, scale) + 34


def pill(image, x, y, label, color=CYAN, active=True, h=24, scale=0.34):
    wd = pill_width(label, scale)
    r = h // 2
    cy = y + h // 2
    if active:
        tint_rrect(image, x, y, x + wd, y + h, r, color, 0.16)
        rrect_line(image, x, y, x + wd, y + h, r, tuple(int(c * 0.62) for c in color), 1)
        cv2.circle(image, (x + 13, cy), 6, tuple(int(c * 0.38) for c in color), -1, AA)
    else:
        tint_rrect(image, x, y, x + wd, y + h, r, (70, 76, 46), 0.28)
        rrect_line(image, x, y, x + wd, y + h, r, (60, 66, 40), 1)
    cv2.circle(image, (x + 13, cy), 3, color if active else INK3, -1, AA)
    txt(image, label, (x + 24, cy + 4), scale, INK1 if active else INK3, 1, shadow=False)
    return wd


def keycap_width(key, label) -> int:
    return max(22, tw(key, 0.34) + 14) + 8 + tw(label, 0.34)


def keycap(image, x, y, key, label, h=22) -> int:
    kw = max(h, tw(key, 0.34) + 14)
    tint_rrect(image, x, y, x + kw, y + h, 6, (86, 92, 48), 0.50)
    rrect_line(image, x, y, x + kw, y + h, 6, HAIR2, 1)
    txt(image, key, (x + (kw - tw(key, 0.34)) // 2, y + h // 2 + 4), 0.34, INK1, 1, shadow=False)
    txt(image, label, (x + kw + 8, y + h // 2 + 4), 0.34, INK3, 1, shadow=False)
    return kw + 8 + tw(label, 0.34)


def ring_gauge(image, cx, cy, r, val, label, color=CYAN):
    val = float(np.clip(val, 0.0, 1.0))
    cv2.ellipse(image, (cx, cy), (r, r), 0, 135, 405, (52, 58, 32), 5, AA)
    if val > 0.01:
        cv2.ellipse(image, (cx, cy), (r + 2, r + 2), 0, 135, 135 + 270 * val,
                    tuple(int(c * 0.35) for c in color), 2, AA)
        cv2.ellipse(image, (cx, cy), (r, r), 0, 135, 135 + 270 * val, color, 5, AA)
        a = math.radians(135 + 270 * val)
        cv2.circle(image, (int(cx + r * math.cos(a)), int(cy + r * math.sin(a))), 4, WHITE, -1, AA)
    s = str(int(round(val * 100)))
    txt(image, s, (cx - tw(s, 0.55, FONT_D) // 2, cy + 7), 0.55, INK1, 1, FONT_D, False)
    lw = tracked_width(label, 0.27, 1)
    put_tracked(image, label, (cx - lw // 2, cy + r + 17), 0.27, INK3, 1)


def bipolar(image, x0, x1, y, val, rng, color=CYAN):
    """Centre-zero slider with a glowing knob."""
    mid = (x0 + x1) // 2
    tint_rrect(image, x0, y - 2, x1, y + 3, 2, (62, 68, 40), 0.95)
    cv2.line(image, (mid, y - 6), (mid, y + 7), HAIR2, 1, AA)
    k = float(np.clip(val / rng, -1, 1))
    xe = int(mid + k * (x1 - x0) / 2)
    if xe != mid:
        cv2.line(image, (min(mid, xe), y), (max(mid, xe), y), color, 3, AA)
    cv2.circle(image, (xe, y), 7, tuple(int(c * 0.30) for c in color), -1, AA)
    cv2.circle(image, (xe, y), 4, (14, 16, 6), -1, AA)
    cv2.circle(image, (xe, y), 4, color, 1, AA)
    cv2.circle(image, (xe, y), 1, WHITE, -1, AA)


def chat_button(image, x, y):
    wd, ht = 130, 34
    tint_rrect(image, x, y, x + wd, y + ht, 17, CYAN, 0.20)
    rrect_line(image, x, y, x + wd, y + ht, 17, CYAN, 1)
    cv2.line(image, (x + 21, y + 17), (x + 31, y + 17), CYAN, 2, AA)
    cv2.line(image, (x + 26, y + 12), (x + 26, y + 22), CYAN, 2, AA)
    put_tracked(image, "OPEN CHAT", (x + 40, y + 22), 0.32, INK1, 2)
    return (x, y, x + wd, y + ht)


def chat_bubble(image, x0, y0, x1, y1, label, accent, strong=False):
    tint_rrect(image, x0, y0, x1, y1, 10, accent, 0.17 if strong else 0.09)
    rrect_line(image, x0, y0, x1, y1, 10, tuple(int(c * (0.9 if strong else 0.5)) for c in accent), 1)
    cv2.line(image, (x0 + 1, y0 + 11), (x0 + 1, y1 - 11), accent, 2, AA)
    put_tracked(image, label, (x0 + 16, y0 + 19), 0.28, accent, 2)


_VIG: dict = {}


def video_background(frame, w, h, dim):
    if frame is None:
        return np.zeros((h, w, 3), np.uint8)
    if frame.shape[1] != w or frame.shape[0] != h:
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gain = float(np.clip(70.0 / max(float(gray.mean()), 1.0), 0.12, 1.0))
    gray = np.clip(gray * gain, 0, 255) * dim
    out = np.dstack([gray * 0.95, gray, gray * 0.30]) * 0.75 + frame.astype(np.float32) * gain * dim * 0.35
    out[::3] *= 0.80
    if (w, h) not in _VIG:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.sqrt(((xx - w / 2) / (w * 0.75)) ** 2 + ((yy - h / 2) / (h * 0.75)) ** 2)
        _VIG[(w, h)] = np.clip(1.15 - r * 0.6, 0.35, 1.0)[..., None]
    out *= _VIG[(w, h)]
    return np.clip(out, 0, 255).astype(np.uint8)


def studio_background(w, h):
    """Deep navy stage with a soft central light pillar."""
    key = ("studio", w, h)
    if key not in _VIG:
        ys = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
        bg = np.array([25, 20, 15], np.float32) * (1 - ys) + np.array([13, 11, 9], np.float32) * ys
        bg = np.broadcast_to(bg, (h, w, 3)).copy()
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.sqrt(((xx - w / 2) / (w * 0.56)) ** 2 + ((yy - h * 0.48) / (h * 0.74)) ** 2)
        bg *= np.clip(1.28 - r * 0.70, 0.30, 1.12)[..., None]
        halo = np.exp(-(((xx - w * 0.50) / (w * 0.30)) ** 2
                        + ((yy - h * 0.50) / (h * 0.52)) ** 2) * 2.2)
        bg += halo[..., None] * np.array((30, 18, 8), np.float32)
        pillar = np.exp(-(((xx - w * 0.50) / (w * 0.085)) ** 2)) * np.clip(1.0 - yy / h * 0.55, 0, 1)
        bg += pillar[..., None] * np.array((25, 18, 8), np.float32)          # faint light pillar
        floor = np.exp(-(((yy - h * 0.86) / (h * 0.10)) ** 2)) * np.exp(-(((xx - w * 0.5) / (w * 0.34)) ** 2))
        bg += floor[..., None] * np.array((40, 34, 14), np.float32)         # base glow on the floor
        _VIG[key] = np.clip(bg, 0, 255).astype(np.uint8)
    return _VIG[key].copy()


# --------------------------------------------------------------------------- voice (listen + talk)
CPS = 38.0           # robot reply typing speed (characters per second)

SPEECH_LANGUAGES = {"english": "en-IN"}


def resolve_speech_language(value: str) -> str:
    language = value.strip().casefold()
    if language in SPEECH_LANGUAGES:
        return SPEECH_LANGUAGES[language]
    if language.startswith("en-"):
        return value.strip()
    return "en-IN"


def speech_language_name(locale: str) -> str:
    if locale.casefold().startswith("en-"):
        return "English"
    return next((name.title() for name, code in SPEECH_LANGUAGES.items()
                 if code.casefold() == locale.casefold()), "English")


def reply_language_instruction(locale: str) -> str:
    return "Write every part of your answer in English. Use English words and Latin letters only."


def response_uses_expected_script(text: str, locale: str) -> bool:
    code = resolve_speech_language(locale)
    return code.startswith("en-") and all(ord(char) < 128 for char in text)


def fallback_reply(text: str) -> str:
    s = " " + re.sub(r"[^a-z0-9 ]", " ", text.lower()) + " "
    question = re.match(r"\s*(what|who|where|when|why|how|which|explain|tell me)\b", text.lower())
    if question:
        aliases = {
            "face": "HEAD", "finger": "HAND", "fingers": "HAND", "arm": "UPPER ARM",
            "leg": "THIGH", "legs": "THIGH", "waist": "ABDOMEN",
        }
        for name in sorted(PART_INFO, key=len, reverse=True):
            if f" {name.lower()} " in s:
                return PART_INFO[name]
        for alias, name in aliases.items():
            if f" {alias} " in s:
                return PART_INFO[name]
    if any(k in s for k in (" hello ", " hi ", " hey ", " good morning ", " good evening ", " namaste ")):
        return f"Hello! I am {ROBOT_NAME}. Ask me about my parts, or tell me to explode, zoom or rotate."
    if " your name" in s or " who are you" in s:
        return (f"I am {ROBOT_NAME}, a virtual humanoid robot. "
                "I can follow your head and hands and answer your questions.")
    if " how are you" in s:
        return "I am ready to talk and respond to your controls. Thanks for asking!"
    if " what can you do" in s or " help" in s:
        return ("I follow your head and hands. Say explode, assemble, zoom in, rotate left, "
                "or ask me to show the knee, the hand or the head.")
    if " thank" in s:
        return "You are welcome!"
    if " bye" in s or " goodbye" in s:
        return "Goodbye! Say hello when you want to talk again."
    if " time " in s:
        return "It is " + time.strftime("%I:%M %p") + "."
    return ("I could not generate an answer just now. Please try again. "
            "I can still answer simple questions and follow robot controls such as explode, "
            "show the knee, and zoom in.")


class Speaker:
    """Text to speech in a background thread (macOS say / pyttsx3 / espeak)."""

    def __init__(self, enabled: bool = True, lang: str = "en-IN"):
        self.q: "queue.Queue[str]" = queue.Queue()
        self.busy = False
        self.backend: Optional[str] = None
        self.proc = None
        self.lang = resolve_speech_language(lang)
        self.voice_name: Optional[str] = None
        self.rate = 165
        self.status = "voice output off"
        if not enabled:
            return
        if sys.platform == "darwin" and shutil.which("say"):
            self.backend = "say"
            self.voice_name = self._find_say_voice(self.lang)
        else:
            try:
                import pyttsx3  # noqa: F401
                self.backend = "pyttsx3"
            except Exception:
                for exe in ("espeak-ng", "espeak", "spd-say"):
                    if shutil.which(exe):
                        self.backend = exe
                        break
        if self.backend:
            if self.backend == "say":
                self._update_voice_status()
            else:
                self.status = f"robot voice: {self.backend}"
            threading.Thread(target=self._run, daemon=True).start()
        else:
            self.status = "robot is silent (pip install pyttsx3 for a voice)"

    @staticmethod
    def _available_say_voices() -> list[tuple[str, str]]:
        try:
            result = subprocess.run(["say", "-v", "?"], capture_output=True, text=True,
                                    timeout=5, check=False)
        except (OSError, subprocess.TimeoutExpired):
            return []
        voices: list[tuple[str, str]] = []
        for line in result.stdout.splitlines():
            match = re.match(r"\s*(.+?)\s+([a-z]{2,3}[_-][A-Z]{2,4})\s+#", line)
            if match:
                voices.append((match.group(1).strip(), match.group(2).replace("_", "-")))
        return voices

    @classmethod
    def _find_say_voice(cls, lang: str) -> Optional[str]:
        wanted = lang.split("-", 1)[0].casefold()
        voices = [name for name, locale in cls._available_say_voices()
                  if locale.split("-", 1)[0].casefold() == wanted]
        if not voices:
            return None
        preferred = {
            "en": ("Samantha", "Ava", "Allison", "Alex"),
            "es": ("Monica", "Paulina", "Jorge"),
            "fr": ("Amelie", "Thomas"),
            "de": ("Anna", "Yannick"),
        }.get(wanted, ())
        return next((name for choice in preferred for name in voices
                     if name.casefold() == choice.casefold()), voices[0])

    def _update_voice_status(self):
        language = speech_language_name(self.lang)
        if self.voice_name:
            self.status = f"robot voice: {self.voice_name} ({self.rate} wpm)"
        else:
            self.status = f"no installed {language} voice; macOS default voice will be used"

    def set_language(self, lang: str):
        self.lang = resolve_speech_language(lang)
        if self.backend == "say":
            self.voice_name = self._find_say_voice(self.lang)
            self._update_voice_status()

    def speak(self, text: str):
        if not self.backend:
            return
        self.stop()
        self.q.put(text)

    def stop(self):
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass
        p = self.proc
        if p is not None and p.poll() is None:
            p.terminate()

    def _run(self):
        engine = None
        while True:
            text = self.q.get()
            self.busy = True
            try:
                if self.backend == "pyttsx3":
                    if engine is None:
                        import pyttsx3
                        engine = pyttsx3.init()
                        engine.setProperty("rate", self.rate)
                    engine.say(text)
                    engine.runAndWait()
                else:
                    if self.backend == "say" and self.voice_name:
                        cmd = [self.backend, "-v", self.voice_name, "-r", str(self.rate), text]
                    elif self.backend == "spd-say":
                        cmd = [self.backend, "-w", text]
                    elif self.backend in ("espeak", "espeak-ng"):
                        cmd = [self.backend, "-s", str(self.rate), text]
                    else:
                        cmd = [self.backend, text]
                    self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    self.proc.wait()
            except Exception:
                pass
            finally:
                self.busy = False


class Voice:
    """Microphone -> text (SpeechRecognition) and robot replies (Ollama or built-in rules)."""

    def __init__(self, guide: OllamaGuide, speaker: Speaker, enabled: bool = True,
                 lang: str = "en-IN", mic_index: Optional[int] = None):
        self.guide = guide
        self.speaker = speaker
        self.lang = resolve_speech_language(lang)
        self.mic_index = mic_index
        self.inbox: "queue.Queue[str]" = queue.Queue()
        self.log: collections.deque = collections.deque(maxlen=14)
        self.history: list = []
        self.level = 0.0
        self.muted = False
        self.mic_ok = False
        self.hearing = False
        self.recognizing = False
        self.thinking = False
        self.heard = ""
        self.heard_t = 0.0
        self.reply = ""
        self.reply_t = 0.0
        self.ai_status = guide.status
        self.mute_until = 0.0
        self.status = "microphone off (--no-voice) - press Y to type to the robot"
        if enabled:
            self.status = "starting microphone..."
            threading.Thread(target=self._listen, daemon=True).start()

    # ---- state helpers (read by the UI)
    def typing_active(self) -> bool:
        return bool(self.reply) and (time.time() - self.reply_t) * CPS < len(self.reply)

    def talking(self) -> bool:
        return self.speaker.busy or self.typing_active()

    def shown_reply(self) -> str:
        if not self.reply.isascii():
            return self.reply
        n = int((time.time() - self.reply_t) * CPS)
        return self.reply[:max(n, 0)]

    def state(self) -> str:
        if self.talking():
            return "SPEAKING"
        if self.thinking:
            return "THINKING"
        if self.recognizing:
            return "RECOGNIZING"
        if self.hearing:
            return "HEARING YOU"
        if self.muted:
            return "MUTED"
        if self.mic_ok:
            return "LISTENING"
        return "TYPE MODE"

    # ---- conversation
    def submit(self, text: str):
        text = text.strip()
        if not text:
            return
        if not text.isascii():
            self.status = "English only: please type or speak your message in English."
            return
        self.heard, self.heard_t = text, time.time()
        self.log.append(("YOU", text))
        self.history = (self.history + [("user", text)])[-8:]
        self.inbox.put(text)

    def set_language(self, lang: str):
        self.lang = resolve_speech_language(lang)
        self.speaker.set_language(self.lang)
        self.status = f"speech and reply language: {speech_language_name(self.lang)}"

    def say_command(self, text: str):
        self.say(text)

    def say(self, text: str):
        text = " ".join(text.split())
        if not text:
            return
        self.thinking = False
        self.reply, self.reply_t = text, time.time()
        self.log.append((ROBOT_NAME, text))
        self.history = (self.history + [("assistant", text)])[-8:]
        self.mute_until = time.time() + len(text) / CPS + 0.7
        self.speaker.speak(text)

    def ask(self, text: str, context: str = ""):
        self.thinking = True
        threading.Thread(target=self._ask, args=(text, context), daemon=True).start()

    def _ask(self, text: str, context: str):
        reply = None
        lang = self.lang
        if self.guide.enabled:
            reply_language = speech_language_name(lang)
            self.ai_status = f"asking {self.guide.model} in {reply_language}"
            system = (f"You are {ROBOT_NAME}, a friendly white humanoid robot shown as a 3D hologram. You see the user "
                      "through a webcam and copy their head and hand movements. Answer the user's question "
                      "directly, or respond naturally to their conversation. Reply in at most 50 words, "
                      f"plain text only, no markdown, no emojis. {reply_language_instruction(lang)} "
                      "Do not "
                      "claim to have physical parts or take actions that the app cannot perform. " + context)
            msgs = [{"role": "system", "content": system}] + \
                   [{"role": r, "content": c} for r, c in self.history[-7:]]
            body = json.dumps({"model": self.guide.model, "messages": msgs, "stream": False, "think": False,
                               "options": {"temperature": 0.6, "num_predict": 120}}).encode()
            req = urllib.request.Request(self.guide.url + "/api/chat", data=body,
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    reply = json.load(r).get("message", {}).get("content", "").strip()
                reply = reply.replace("**", "").replace("#", "").replace("`", "").strip()
                if reply and not response_uses_expected_script(reply, lang):
                    self.ai_status = f"model answered outside {reply_language}; retrying in English"
                    reply = self._request_translation(reply, lang)
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                self.ai_status = f"Ollama chat failed ({type(exc).__name__}: {exc}); built-in fallback used"
                reply = None
        if reply:
            self.ai_status = f"answered in {speech_language_name(lang)}"
            self.say(reply)
            return
        fallback = fallback_reply(text)
        if self.guide.enabled and not reply:
            if self.ai_status.startswith("asking "):
                self.ai_status = "Ollama returned no answer; built-in fallback used"
        else:
            self.ai_status = "AI disabled; used built-in reply"
        self.say(fallback)

    def _request_translation(self, text: str, lang: str) -> Optional[str]:
        language = speech_language_name(lang)
        body = json.dumps({
            "model": self.guide.model,
            "messages": [
                {"role": "system", "content": (
                    f"Translate the following answer into {language}. "
                    f"{reply_language_instruction(lang)} Preserve all important facts. "
                    "Return only the translation."
                )},
                {"role": "user", "content": text},
            ],
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 180},
        }).encode()
        req = urllib.request.Request(self.guide.url + "/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                translated = json.load(response).get("message", {}).get("content", "").strip()
            translated = translated.replace("**", "").replace("#", "").replace("`", "").strip()
            if not translated or not response_uses_expected_script(translated, lang):
                raise ValueError(f"Ollama did not return a {language} answer")
            return translated
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            self.status = f"{language} answer unavailable ({type(exc).__name__}); check Ollama/model"
            self.ai_status = self.status
            return None

    # ---- microphone thread: own voice-activity detector so we can show a real level meter
    def _listen(self):
        try:
            import speech_recognition as sr
        except ImportError:
            self.status = "mic off: pip install SpeechRecognition pyaudio   (press Y to type instead)"
            return
        try:
            mic = sr.Microphone(device_index=self.mic_index, sample_rate=16000)
            rec = sr.Recognizer()
            src = mic.__enter__()
        except Exception as exc:
            self.status = f"mic unavailable ({type(exc).__name__}): install pyaudio / allow the microphone"
            return
        self.mic_ok = True
        self.status = "listening - speak to the robot"
        chunk = mic.CHUNK
        pre: collections.deque = collections.deque(maxlen=6)
        frames: list = []
        rec_on, silence, voiced, warm, noise = False, 0, 0, 0, 300.0
        try:
            while True:
                data = src.stream.read(chunk)
                arr = np.frombuffer(data, np.int16).astype(np.float32)
                rms = float(np.sqrt(np.mean(arr * arr))) if len(arr) else 0.0
                self.level += (min(rms / 4000.0, 1.0) - self.level) * 0.5
                now = time.time()
                if self.muted or self.speaker.busy or now < self.mute_until or self.thinking:
                    pre.clear()
                    frames, rec_on, silence, voiced, self.hearing = [], False, 0, 0, False
                    continue
                if warm < 15:                                   # measure the room noise first
                    noise = 0.8 * noise + 0.2 * rms
                    warm += 1
                    continue
                loud = rms > max(noise * 2.6, 450.0)
                if not rec_on:
                    pre.append(data)
                    if loud:
                        rec_on, frames, voiced, silence, self.hearing = True, list(pre), 1, 0, True
                    else:
                        noise = 0.97 * noise + 0.03 * rms
                else:
                    frames.append(data)
                    if loud:
                        voiced, silence = voiced + 1, 0
                    else:
                        silence += 1
                    if silence >= 13 or len(frames) > 160:      # ~0.8 s of silence or 10 s max
                        self.hearing, rec_on = False, False
                        if voiced >= 4:
                            audio = sr.AudioData(b"".join(frames), 16000, 2)
                            threading.Thread(target=self._recognize, args=(sr, rec, audio),
                                             daemon=True).start()
                        frames, voiced, silence = [], 0, 0
                        pre.clear()
        except Exception as exc:
            self.mic_ok = False
            self.status = f"mic stopped: {exc}"

    def _recognize(self, sr, rec, audio):
        self.recognizing = True
        text = ""
        try:
            text = rec.recognize_google(audio, language=self.lang).strip()
            self.status = "listening - speak to the robot"
        except sr.UnknownValueError:
            self.status = "could not understand - please say it again"
        except sr.RequestError:
            self.status = "speech service unreachable (needs internet) - press Y to type"
        except Exception as exc:
            self.status = f"recognition error: {exc}"
        finally:
            self.recognizing = False
        if text:
            self.submit(text)


# --------------------------------------------------------------------------- application
CAM_BOX, CAM_FULL, CAM_OFF = "box", "full", "off"
BOX_W, BOX_H = 260, 146


class App:
    def __init__(self, args, guide, voice=None):
        self.args = args
        self.guide = guide
        self.voice = voice
        self.typing = False
        self.typed = ""
        self.want_shot = False
        self.want_chat = False
        self.chat_button_bounds = (0, 0, 0, 0)
        self.wave: collections.deque = collections.deque([0.0] * 56, maxlen=56)
        self.w, self.h = args.width, args.height
        self.view = View(self.w, self.h)
        self.robot = RobotModel()
        logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "images", "IMG_0977.png")
        logo = cv2.imread(logo_path, cv2.IMREAD_COLOR)
        if logo is None:
            raise FileNotFoundError(f"Unable to load hologram background logo: {logo_path}")
        logo_distance = 255.0 - logo.min(axis=2).astype(np.float32)
        logo_alpha = np.clip((logo_distance - 10.0) * 4.0, 0, 255).astype(np.uint8)
        logo_y, logo_x = np.where(logo_alpha > 10)
        if not len(logo_x):
            raise ValueError(f"Header logo has no visible foreground: {logo_path}")
        logo = logo[logo_y.min():logo_y.max() + 1, logo_x.min():logo_x.max() + 1]
        logo_alpha = logo_alpha[logo_y.min():logo_y.max() + 1, logo_x.min():logo_x.max() + 1]
        logo_h, logo_w = logo_alpha.shape
        scale = min(136.0 / logo_w, 46.0 / logo_h)
        logo_size = (max(1, int(logo_w * scale)), max(1, int(logo_h * scale)))
        self.logo_color = cv2.resize(logo, logo_size, interpolation=cv2.INTER_AREA)
        self.logo_alpha = cv2.resize(logo_alpha, logo_size, interpolation=cv2.INTER_AREA)
        self.motion = None
        self.motion_side = "right"
        self.motion_started = 0.0
        self.pose_action = None
        self.pose_side = "right"
        self.head_turn_started = None
        self.explode = 0.0
        self.explode_goal = 0.0
        self.selected = -1
        self.hover = -1
        self.regions = []           # (name, sx, sy, radius_px, depth)
        self.part_info = []
        self.hands = {}
        self.face = None
        self.mini = None            # cropped wireframe of the robot for the LIVE SCAN card
        self.stats = (0, 0)         # faces, vertices
        self.head_pose = (0.0, 0.0, 0.0)       # yaw, pitch(down +), roll in degrees (calibrated)
        self.cal = None                        # neutral (yaw_r, pitch_r, roll)
        self.cal_acc = []
        if args.no_camera or args.no_video:
            self.cam_mode = CAM_OFF
        elif args.full_video:
            self.cam_mode = CAM_FULL
        else:
            self.cam_mode = CAM_BOX
        self.show_hands = True
        self.prev_pinch = {"left": False, "right": False}
        self.drag = None            # (side, last pointer)
        self.prev_span = None
        self.fist_frames = 0
        self.open_frames = 0
        self.clap_cool = 0.0
        self.m_down = None
        self.m_last = None
        self.m_moved = False
        rs = np.random.RandomState(7)
        self.pn = 48                                   # restrained ambient hologram particles
        self.p_ang = rs.uniform(0, 2 * math.pi, self.pn)
        self.p_rad = rs.uniform(0.18, 1.10, self.pn)
        self.p_y = rs.uniform(0, 1, self.pn)
        self.p_spd = rs.uniform(0.05, 0.16, self.pn)

    # ---- selection helpers
    def hit(self, p):
        best, best_ratio = -1, 1.0
        for i, (_, sx, sy, r, _z) in enumerate(self.regions):
            k = math.hypot(p[0] - sx, p[1] - sy) / max(r, 1.0)
            if k < best_ratio:
                best, best_ratio = i, k
        return best

    def hit_chat_button(self, x, y):
        x0, y0, x1, y1 = self.chat_button_bounds
        return x0 <= x <= x1 and y0 <= y <= y1

    def open_chat(self):
        self.typing = True
        self.typed = ""

    def handle_chat_key(self, key):
        if not self.typing:
            return False
        if key in (10, 13):
            message = self.typed.strip()
            self.typing = False
            self.typed = ""
            if message and self.voice is not None:
                self.voice.submit(message)
            return True
        if key == 27:
            self.typing = False
            self.typed = ""
            return True
        if key in (8, 127):
            self.typed = self.typed[:-1]
            return True
        if 32 <= key <= 126 and len(self.typed) < 500:
            self.typed += chr(key)
            return True
        return True

    def select(self, idx, force=False):
        if idx < 0:
            return
        self.selected = idx
        self.guide.request(PART_NAMES[idx], force)

    def cycle(self, step):
        self.select((self.selected + step) % len(PART_NAMES) if self.selected >= 0 else 0)

    def focus(self):
        if self.selected >= 0 and self.part_info:
            c, r = self.part_info[self.selected]
            self.view.T_goal = np.asarray(c, np.float32).copy()
            self.view.zoom_goal = float(np.clip(0.55 / max(r, 0.05) * 0.45 + 1.2, 1.5, 6.0))

    def reset_view(self):
        v = self.view
        v.zoom_goal = 1.0
        v.T_goal = v3(0, .9 + 0.4 * self.explode_goal, 0)
        v.yaw, v.pitch, v.auto = 0.0, 0.05, False

    def cycle_cam(self):
        if self.args.no_camera:
            return
        order = [CAM_BOX, CAM_FULL, CAM_OFF]
        self.cam_mode = order[(order.index(self.cam_mode) + 1) % 3]

    def calibrate(self):
        self.cal = None
        self.cal_acc = []

    def set_pose(self, action, side="right"):
        self.motion = None
        self.pose_action = action
        self.pose_side = side
        self.motion_started = time.monotonic()

    # ---- per-frame logic
    def step(self, t, dt, hands, face=None):
        self.hands = hands
        self.face = face
        rb, view = self.robot, self.view

        if len(hands) == 2 and all(h["count"] == 0 for h in hands.values()):
            self.fist_frames += 1
        else:
            self.fist_frames = 0
        if self.fist_frames >= 12:
            rb.standby = True
        if rb.standby and any(h["count"] >= 4 for h in hands.values()):
            self.open_frames += 1
            if self.open_frames >= 6:
                rb.standby = False
        else:
            self.open_frames = 0

        # clap (two open hands together) toggles explode
        self.clap_cool = max(0.0, self.clap_cool - dt)
        if len(hands) == 2 and all(h["count"] >= 4 for h in hands.values()):
            gap = float(np.hypot(*(hands["left"]["wrist"] - hands["right"]["wrist"])))
            if gap < 0.13 * self.w and self.clap_cool == 0.0:
                self.toggle_explode()
                self.clap_cool = 1.5

        # pinch gestures: select / rotate / two-hand zoom
        pinching = [s for s, h in hands.items() if h["pinch"]]
        if len(pinching) == 2:
            span = float(np.hypot(*(hands["left"]["pointer"] - hands["right"]["pointer"])))
            if self.prev_span:
                view.zoom_goal *= span / self.prev_span
            self.prev_span = span
            self.drag = None
        else:
            self.prev_span = None
            for side, h in hands.items():
                if h["pinch"] and not self.prev_pinch[side] and len(pinching) == 1:
                    idx = self.hit(h["pointer"])
                    if idx >= 0:
                        self.select(idx)
                    else:
                        self.drag = (side, h["pointer"].copy())
                if self.drag and self.drag[0] == side:
                    if h["pinch"]:
                        dx, dy = h["pointer"] - self.drag[1]
                        view.yaw -= float(dx) * 0.010
                        view.pitch = float(np.clip(view.pitch + float(dy) * 0.005, -1.1, 1.1))
                        view.auto = False
                        self.drag = (side, h["pointer"].copy())
                    else:
                        self.drag = None
        if self.drag and self.drag[0] not in hands:
            self.drag = None
        self.prev_pinch = {s: (s in hands and hands[s]["pinch"]) for s in ("left", "right")}

        self.hover = -1
        for h in hands.values():
            i = self.hit(h["pointer"])
            if i >= 0:
                self.hover = i
                break

        # arm / finger targets (robot-local space)
        rb.idle_pose = not hands
        for side in ("left", "right"):
            h = hands.get(side)
            live = h is not None and not rb.standby
            rb.arm_goal[side] = (view.unproject_local(float(h["wrist"][0]), float(h["wrist"][1]))
                                 if live else None)
            rb.curl_goal[side] = h["curls"] if live else [0.2] * 5

        # ---- head: follows YOUR face (webcam); falls back to looking at your fingertips
        rb.face_lock = face is not None and not rb.standby
        if face is not None:
            if self.cal is None:                       # auto-calibrate on the first frames
                self.cal_acc.append((face["yaw_r"], face["pitch_r"], face["roll"]))
                if len(self.cal_acc) >= 20:
                    self.cal = tuple(np.mean(self.cal_acc, axis=0))
                cal = (face["yaw_r"], face["pitch_r"], face["roll"])
            else:
                cal = self.cal
            yaw = float(np.clip((face["yaw_r"] - cal[0]) * 3.6, -1.1, 1.1))
            pit = float(np.clip((face["pitch_r"] - cal[1]) * 3.2, -0.6, 0.6))
            rol = float(np.clip(face["roll"] - cal[2], -0.6, 0.6))
            self.head_pose = (math.degrees(yaw), math.degrees(pit), math.degrees(rol))
            if not rb.standby:
                # screen-right turn must turn the robot head toward screen-right = negative yaw
                rb.look_yaw, rb.look_nod, rb.look_roll = -yaw, -pit, -rol
                rb.lean_goal = float(np.clip((face["center"][0] - 0.5) * 0.30, -0.10, 0.10))
        else:
            self.head_pose = (0.0, 0.0, 0.0)
            if hands and not rb.standby:
                tip = np.mean([h["tip"] for h in hands.values()], axis=0)
                hs = self.head_screen()
                rb.look_yaw = -float(np.clip((tip[0] - hs[0]) / 450.0, -1, 1)) * 0.8
                rb.look_nod = -float(np.clip((tip[1] - hs[1]) / 700.0, -0.3, 0.3))
                rb.look_roll = 0.0
                xs = [h["wrist"][0] for h in hands.values()]
                rb.lean_goal = float(np.clip((np.mean(xs) / self.w - 0.5) * 0.3, -0.12, 0.12))
            else:
                rb.lean_goal, rb.look_yaw, rb.look_nod, rb.look_roll = 0.0, 0.0, 0.0, 0.0
        if rb.standby:
            rb.lean_goal, rb.look_yaw, rb.look_nod, rb.look_roll = 0.0, 0.0, 0.0, 0.0

        if self.motion and not rb.standby:
            elapsed = time.monotonic() - self.motion_started
            rb.idle_pose = False
            if self.motion == "dance":
                beat = elapsed * 2.4
                rb.arm_goal["left"] = v3(-0.34, 1.64 + 0.12 * math.sin(beat),
                                         -0.05 + 0.06 * math.cos(beat))
                rb.arm_goal["right"] = v3(0.34, 1.64 + 0.12 * math.sin(beat + math.pi),
                                          -0.05 - 0.06 * math.cos(beat))
                rb.curl_goal["left"] = rb.curl_goal["right"] = [0.15] * 5
                rb.lean_goal = 0.08 * math.sin(beat)
                rb.look_yaw = 0.22 * math.sin(beat * 0.5)
                rb.look_nod = 0.08 * math.cos(beat)
                rb.look_roll = 0.0
                rb.leg_thigh_goal["left"] = 0.22 + 0.32 * max(0.0, math.sin(beat))
                rb.leg_thigh_goal["right"] = 0.22 + 0.32 * max(0.0, math.sin(beat + math.pi))
                rb.leg_knee_goal["left"] = 0.20 + 0.30 * max(0.0, math.sin(beat))
                rb.leg_knee_goal["right"] = 0.20 + 0.30 * max(0.0, math.sin(beat + math.pi))
            elif self.motion in ("raise", "wave"):
                phase = elapsed * 5.0
                for side in ("left", "right"):
                    active = self.motion_side == "both" or self.motion_side == side
                    if active:
                        sg = -1 if side == "left" else 1
                        sway = 0.07 * math.sin(phase) if self.motion == "wave" else 0.0
                        rb.arm_goal[side] = v3(sg * (0.30 + sway), 1.78 + 0.025 * math.sin(phase),
                                               -0.04)
                        rb.curl_goal[side] = [0.05] * 5
                    else:
                        rb.arm_goal[side] = None

        if not rb.standby:
            rb.body_drop_goal = 0.0
            rb.body_pitch_goal = 0.0
            if self.motion != "dance":
                for side in ("left", "right"):
                    rb.leg_thigh_goal[side] = 0.0
                    rb.leg_knee_goal[side] = 0.0
            for side in ("left", "right"):
                rb.leg_splay_goal[side] = 0.0
            action = self.pose_action
            elapsed = time.monotonic() - self.motion_started
            if action == "jump" and elapsed >= 1.0:
                self.pose_action = None
                action = None
            if action == "sit":
                rb.body_drop_goal = 0.18
                for side in ("left", "right"):
                    rb.leg_thigh_goal[side] = 1.15
                    rb.leg_knee_goal[side] = 1.15
                    sg = -1 if side == "left" else 1
                    rb.arm_goal[side] = v3(sg * .13, 0.88, -.18)
            elif action == "leg_lift":
                side = self.pose_side
                rb.leg_thigh_goal[side] = 2.0
                rb.leg_knee_goal[side] = 1.2
            elif action == "squat":
                rb.body_drop_goal = 0.07
                for side in ("left", "right"):
                    rb.leg_thigh_goal[side] = 0.28
                    rb.leg_knee_goal[side] = 0.72
            elif action == "march":
                phase = (time.monotonic() - self.motion_started) * 4.0
                for i, side in enumerate(("left", "right")):
                    lift = max(0.0, math.sin(phase + i * math.pi))
                    rb.leg_thigh_goal[side] = 1.65 * lift
                    rb.leg_knee_goal[side] = 1.15 * lift
                    sg = -1 if side == "left" else 1
                    rb.arm_goal[side] = v3(sg * .28, 1.28 + .15 * math.sin(phase + (i + 1) * math.pi),
                                           -.02)
            elif action == "tree":
                rb.leg_thigh_goal[self.pose_side] = 2.0
                rb.leg_knee_goal[self.pose_side] = 1.2
                for side in ("left", "right"):
                    sg = -1 if side == "left" else 1
                    rb.arm_goal[side] = v3(sg * .11, 1.88, -.01)
                    rb.curl_goal[side] = [0.05] * 5
            elif action == "warrior":
                rb.leg_thigh_goal["left"] = 0.18
                rb.leg_thigh_goal["right"] = 0.18
                rb.leg_knee_goal["left"] = 0.35
                rb.leg_knee_goal["right"] = 0.35
                rb.arm_goal["left"] = v3(-.48, 1.43, -.01)
                rb.arm_goal["right"] = v3(.48, 1.43, -.01)
                rb.curl_goal["left"] = rb.curl_goal["right"] = [0.05] * 5
            elif action == "stretch":
                rb.arm_goal["left"] = v3(-.12, 1.88, -.01)
                rb.arm_goal["right"] = v3(.12, 1.88, -.01)
            elif action == "funny":
                rb.arm_goal["left"] = v3(-.34, 1.75, -.02)
                rb.arm_goal["right"] = v3(.34, 1.20, -.18)
                rb.look_roll = 0.28
                rb.curl_goal["left"] = rb.curl_goal["right"] = [0.05] * 5
            elif action == "jump":
                phase = min(elapsed / 1.0, 1.0)
                height = math.sin(math.pi * phase)
                rb.body_drop_goal = -0.24 * height
                rb.leg_thigh_goal["left"] = rb.leg_thigh_goal["right"] = 0.38 * height
                rb.leg_knee_goal["left"] = rb.leg_knee_goal["right"] = 0.55 * height
                rb.arm_goal["left"] = v3(-.31, 1.62 + .18 * height, -.03)
                rb.arm_goal["right"] = v3(.31, 1.62 + .18 * height, -.03)
            elif action == "jumping_jack":
                phase = elapsed * 3.2
                spread = max(0.0, math.sin(phase))
                rb.body_drop_goal = -0.11 * max(0.0, math.sin(phase * 2))
                for side in ("left", "right"):
                    sg = -1 if side == "left" else 1
                    rb.leg_splay_goal[side] = 0.42 * spread
                    rb.leg_thigh_goal[side] = 0.12 * spread
                    rb.arm_goal[side] = v3(sg * (.28 + .10 * spread),
                                           1.40 + .38 * spread, -.02)
                    rb.curl_goal[side] = [0.05] * 5
            elif action == "pushup":
                phase = elapsed * 2.8
                dip = max(0.0, math.sin(phase))
                rb.body_pitch_goal = -1.18
                rb.body_drop_goal = .58 + .08 * dip
                for side in ("left", "right"):
                    sg = -1 if side == "left" else 1
                    rb.arm_goal[side] = v3(sg * .23, 1.04 + .12 * dip, -.10)
                    rb.curl_goal[side] = [0.05] * 5

        talking = self.voice is not None and self.voice.talking()
        rb.mouth_goal = (0.3 + 0.7 * abs(math.sin(t * 15))) if talking else 0.0
        rb.update(t)
        if self.head_turn_started is not None:
            progress = min((time.monotonic() - self.head_turn_started) / 2.5, 1.0)
            rb.head_yaw = 2 * math.pi * progress
            rb.head_nod = rb.head_roll = 0.0
            if progress >= 1.0:
                self.head_turn_started = None
                rb.head_yaw = 0.0
        self.explode += (self.explode_goal - self.explode) * 0.12
        if not (self.selected >= 0 and view.zoom_goal > 1.4):
            view.T_goal[1] = .9 + 0.4 * self.explode
        view.update(dt, self.explode)

    # ---- voice commands (instant, no AI needed)
    PART_WORDS = (("upper arm", "UPPER ARM"), ("forearm", "FOREARM"), ("fore arm", "FOREARM"),
                  ("shoulder", "SHOULDER"), ("elbow", "ELBOW"), ("hand", "HAND"), ("finger", "HAND"),
                  ("hip", "HIP"), ("thigh", "THIGH"), ("knee", "KNEE"), ("shin", "SHIN"),
                  ("foot", "FOOT"), ("feet", "FOOT"), ("leg", "THIGH"), ("arm", "UPPER ARM"),
                  ("head", "HEAD"), ("face", "HEAD"), ("neck", "NECK"), ("chest", "CHEST"),
                  ("abdomen", "ABDOMEN"), ("waist", "ABDOMEN"), ("pelvis", "PELVIS"))

    def handle_command(self, text: str) -> Optional[str]:
        s = " " + re.sub(r"[^a-z0-9 ]", " ", text.lower()) + " "
        has = lambda *keys: any(k in s for k in keys)       # noqa: E731
        view, rb = self.view, self.robot
        asks_name = has(" your name ", " robot name ", " who are you ", " introduce yourself ")
        asks_version = has(" version ", " app version ", " software version ")
        if asks_name or asks_version:
            details = []
            if asks_name:
                details.append(f"{ROBOT_NAME}, version {APP_VERSION}.")
            if asks_version:
                details.append(f"I am running version {APP_VERSION}.")
            return " ".join(details)
        if re.match(r"\s*(what|who|where|when|why|how|which|explain|tell me)\b", text.lower()):
            return None
        if has(" stop exercise", " stop yoga", " stop moving", " be still", " cancel pose"):
            self.motion = None
            self.pose_action = None
            rb.body_drop_goal = 0.0
            return "Okay, I stopped the movement and returned to standing."
        if has(" sit ", " sit down", " take a seat", " sitting"):
            self.set_pose("sit")
            return "Sitting down."
        if has(" stand by", " standby", " sleep", " shut down", " power down"):
            self.motion = None
            self.pose_action = None
            rb.standby = True
            return "Going to standby. Wave an open hand or say wake up."
        if re.search(r"\b(?:stand(?:\s+up|\s+straight)?|get up)\b", s) and not has(" stand by"):
            self.motion = None
            self.pose_action = None
            return "Standing up."
        leg_lift = re.search(
            r"\b(?:lift|raise|kick)\s+(?:(?:up|your|the|my|a)\s+)*(?:(left|right|both)\s+)?legs?\b",
            s
        )
        if leg_lift or has(" left leg up", " right leg up", " lift your leg", " raise your leg"):
            side = (leg_lift.group(1) if leg_lift else None) or (
                "left" if has(" left ") else "right"
            )
            if side == "both":
                self.set_pose("march")
                return "Starting a gentle alternating leg lift."
            self.set_pose("leg_lift", side)
            return f"Lifting my {side} leg. Say lower your leg or stand to reset."
        if has(" lower your leg", " lower the leg", " put your leg down", " legs down",
               " lower your legs"):
            self.motion = None
            self.pose_action = None
            return "Lowering my leg."
        if has(" jumping jack", " jumping jacks"):
            self.set_pose("jumping_jack")
            return "Starting jumping jacks. Say stop exercise to stop."
        if has(" push up", " pushup", " push ups", " pushups", " push up position"):
            self.set_pose("pushup")
            return "Starting a gentle push-up motion. Say stop exercise to stop."
        if re.search(r"\b(?:jump|jumping)\b", s):
            self.set_pose("jump")
            return "Jumping once."
        if has(" yoga", " tree pose", " warrior pose", " meditation"):
            if has(" warrior"):
                self.set_pose("warrior")
                return "Holding a yoga warrior pose."
            if has(" meditation"):
                self.set_pose("sit")
                return "Sitting in a meditation pose."
            self.set_pose("tree")
            return "Balancing in a yoga tree pose."
        if has(" exercise", " work out", " workout", " march in place", " jog in place"):
            self.set_pose("march")
            return "Starting a gentle marching exercise. Say stop exercise to stop."
        if has(" squat", " squats"):
            self.set_pose("squat")
            return "Holding a squat pose. Say stand to return upright."
        if has(" stretch", " stretching", " reach up"):
            self.set_pose("stretch")
            return "Stretching both arms overhead."
        if has(" funny pose", " silly pose", " make a funny pose", " do a joke pose"):
            self.set_pose("funny")
            return "Here's my silly pose."
        if has(" stop rotating your head", " stop turning your head", " stop head rotation",
               " stop rotating head", " stop head turn"):
            self.head_turn_started = None
            rb.head_yaw = rb.look_yaw = 0.0
            return "Okay, I stopped rotating my head."
        head_turn = re.search(
            r"\b(?:rotate|turn|spin)\s+(?:your\s+)?head\b|\bhead\s+(?:360|around)\b", s
        )
        if head_turn:
            self.head_turn_started = time.monotonic()
            return "Turning my head one full 360 degrees."
        if has(" stop dancing", " stop waving", " stop moving", " be still"):
            self.motion = None
            self.pose_action = None
            return "Okay, I stopped and returned to my resting pose."
        if has(" lower your hand", " lower your hands", " lower hand", " put your hand down",
               " put your hands down", " hands down", " arms down"):
            self.motion = None
            self.pose_action = None
            return "Lowering my arms."
        if has(" dance", " dancing"):
            self.motion = "dance"
            self.pose_action = None
            self.motion_side = "both"
            self.motion_started = time.monotonic()
            return "Sure, here's a dance."
        if has(" wave", " say hello", " greet me"):
            self.motion = "wave"
            self.pose_action = None
            self.motion_side = "right"
            if has(" left "):
                self.motion_side = "left"
            elif has(" both ", " hands "):
                self.motion_side = "both"
            self.motion_started = time.monotonic()
            return "Hello! I'm waving."
        raise_hand = re.search(
            r"\b(?:raise|lift)\s+(?:(?:your|a|the|both)\s+)?"
            r"(?:(?:left|right|both)\s+)?(?:hand|hands|arm|arms)\b", s
        )
        if raise_hand or has(" put your hand up", " put your hands up", " hands up", " arms up"):
            self.motion = "raise"
            self.pose_action = None
            plural_target = re.search(r"\b(?:hands|arms)\b", s)
            self.motion_side = "both" if has(" both ") or plural_target else "right"
            if has(" left ") and self.motion_side != "both":
                self.motion_side = "left"
            elif has(" right ") and self.motion_side != "both":
                self.motion_side = "right"
            self.motion_started = time.monotonic()
            side_text = "both hands" if self.motion_side == "both" else f"my {self.motion_side} hand"
            return f"Raising {side_text}."
        if has(" stop spin", " stop rotat", " stop turning"):
            view.auto = False
            return "Okay, I stopped rotating."
        if has(" explode", " separate", " break apart", " take apart", " disassemble", " split"):
            if self.explode_goal < 0.5:
                self.toggle_explode()
            return "Separating every part. Say assemble to put me back together."
        if has(" assemble", " put together", " put me back", " come together", " full robot",
               " reset", " back to normal"):
            self.explode_goal = 0.0
            self.reset_view()
            return "Assembled. Showing the complete robot."
        if has(" zoom in", " closer", " bigger"):
            view.zoom_goal *= 1.5
            return "Zooming in."
        if has(" zoom out", " farther", " smaller"):
            view.zoom_goal /= 1.5
            return "Zooming out."
        if has(" turn around", " show your back", " show me your back", " rotate back"):
            view.yaw += math.pi
            return "Turning around so you can see my back."
        if has(" rotate left", " turn left", " spin left"):
            view.yaw += 0.7
            return "Rotating left."
        if has(" rotate right", " turn right", " spin right"):
            view.yaw -= 0.7
            return "Rotating right."
        if has(" spin", " rotate", " auto rotate", " 360"):
            view.auto = True
            return "Rotating a full 360 degrees. Say stop spinning to stop."
        if has(" wake up", " wake", " activate", " power up"):
            rb.standby = False
            return "I am awake and ready."
        if has(" calibrate", " recalibrate"):
            self.calibrate()
            return "Look straight at the camera. Calibrating your head position."
        if has(" screenshot", " take a picture", " take a photo", " photo"):
            self.want_shot = True
            return "Saving a screenshot."
        for word, kind in self.PART_WORDS:
            if (" " + word + " ") in s or (" " + word + "s ") in s:
                if kind in ("HEAD", "NECK", "CHEST", "ABDOMEN", "PELVIS"):
                    name = kind
                else:
                    name = ("RIGHT " if " right " in s else "LEFT ") + kind
                self.select(PID[name])
                if has(" zoom", " focus", " closer"):
                    self.focus()
                return PART_INFO[kind]
        return None

    def on_speech(self, text: str):
        if self.voice is None:
            return
        reply = self.handle_command(text)
        if reply is not None:
            self.voice.say_command(reply)
        else:
            sel = PART_NAMES[self.selected].lower() if self.selected >= 0 else "nothing"
            self.voice.ask(text, f"Currently the user has selected your {sel}.")

    def head_screen(self):
        if not self.part_info:
            return (self.view.cx, self.view.cy - 200)
        c, _ = self.part_info[PID["HEAD"]]
        sx, sy, _z = self.view.project(np.asarray(c, np.float32)[None])
        return (float(sx[0]), float(sy[0]))

    def toggle_explode(self):
        self.explode_goal = 0.0 if self.explode_goal > 0.5 else 1.0
        # zoom back out so the complete robot / complete exploded set is visible
        self.view.zoom_goal = 1.0
        self.view.T_goal = v3(0, .9 + 0.4 * self.explode_goal, 0)

    # ---- holographic stage (floor grid, glowing projector rings, beams, orbiting orbs, particles)
    def draw_stage(self, canvas, t):
        w, h = self.w, self.h
        view = self.view
        layer = np.zeros((h, w, 3), np.uint8)
        ang = np.linspace(0, 2 * math.pi, 120)

        def ring_pts(rad, y=0.0):
            return np.stack([np.cos(ang) * rad, np.full_like(ang, y), np.sin(ang) * rad], 1).astype(np.float32)

        def poly(P, col, th=1, closed=False):
            sx, sy, z = view.project(P)
            if z.min() > 0.25:
                cv2.polylines(layer, [np.stack([sx, sy], 1).astype(np.int32)], closed, col, th, AA)

        # perspective floor grid
        g = np.linspace(-1.5, 1.5, 21, dtype=np.float32)
        s = np.linspace(-1.5, 1.5, 25, dtype=np.float32)
        lines = []
        for gv in g:
            lines.append(np.stack([s, np.zeros_like(s), np.full_like(s, gv)], 1))
            lines.append(np.stack([np.full_like(s, gv), np.zeros_like(s), s], 1))
        P = np.stack(lines)                                       # (N, 25, 3)
        sx, sy, z = view.project(P)
        gi = np.repeat(np.abs(g) / 1.5, 2)
        for i in range(len(P)):
            if z[i].min() < 0.25:
                continue
            fade = 1.0 - 0.85 * gi[i]
            col = tuple(int(c * fade * 0.50) for c in CYAN_DIM)
            cv2.polylines(layer, [np.stack([sx[i], sy[i]], 1).astype(np.int32)], False, col, 1, AA)

        # projector base: glowing disc + concentric rings + rotating ticks
        sx, sy, z = view.project(ring_pts(0.95))
        if z.min() > 0.25:
            disc = np.zeros((h, w), np.uint8)
            cv2.fillPoly(disc, [np.stack([sx, sy], 1).astype(np.int32)], 255)
            disc = cv2.GaussianBlur(disc, (0, 0), 9)
            layer += (disc[..., None].astype(np.float32) / 255 * np.array((62, 66, 10), np.float32)).astype(np.uint8)
        for rad, col, th in ((0.95, CYAN, 2), (0.80, CYAN_DIM, 1), (0.62, CYAN, 1), (0.40, CYAN_DIM, 1),
                             (0.22, CYAN, 1), (1.12, CYAN_DIM, 1)):
            poly(ring_pts(rad), col, th, True)
        for k in range(32):
            a = k * (2 * math.pi / 32) + t * 0.16
            ln = 0.07 if k % 4 == 0 else 0.035
            seg = np.array([[math.cos(a) * 0.95, 0, math.sin(a) * 0.95],
                            [math.cos(a) * (0.95 - ln), 0, math.sin(a) * (0.95 - ln)]], np.float32)
            sx, sy, z = view.project(seg)
            if z.min() > 0.25:
                cv2.line(layer, (int(sx[0]), int(sy[0])), (int(sx[1]), int(sy[1])), CYAN, 1, AA)
        a0 = (t * 1.6) % (2 * math.pi)
        arc = np.linspace(a0, a0 + 1.1, 16)
        poly(np.stack([np.cos(arc) * 0.95, np.zeros_like(arc), np.sin(arc) * 0.95], 1).astype(np.float32),
             WHITE, 3)
        # outer segmented rings rotating in opposite directions
        for k in range(6):
            a1 = -t * 0.45 + k * math.pi / 3
            arc = np.linspace(a1, a1 + 0.62, 14)
            poly(np.stack([np.cos(arc) * 1.12, np.zeros_like(arc), np.sin(arc) * 1.12], 1).astype(np.float32),
                 CYAN if k % 2 == 0 else CYAN_DIM, 3 if k % 2 == 0 else 2)
        for k in range(4):
            a1 = t * 0.7 + k * math.pi / 2
            arc = np.linspace(a1, a1 + 0.9, 14)
            poly(np.stack([np.cos(arc) * 0.70, np.zeros_like(arc), np.sin(arc) * 0.70], 1).astype(np.float32),
                 CYAN, 2)

        # projector light beams (vertical streaks) + rising scan rings
        top = 1.95
        for k in range(16):
            a = k * (2 * math.pi / 16)
            seg = np.array([[math.cos(a) * 0.62, 0, math.sin(a) * 0.62],
                            [math.cos(a) * 0.55, top, math.sin(a) * 0.55]], np.float32)
            sx, sy, z = view.project(seg)
            if z.min() > 0.25:
                cv2.line(layer, (int(sx[0]), int(sy[0])), (int(sx[1]), int(sy[1])), (22, 21, 17), 1, AA)
        for k in range(3):
            hh = ((t * 0.30 + k / 3.0) % 1.0) * top
            amt = (1.0 - hh / top) * 0.9
            poly(ring_pts(0.60 - 0.05 * hh / top, hh), tuple(int(c * amt) for c in CYAN), 2, True)

        # orbiting energy orbs with their orbit paths
        orbs = ((0.78, 0.35, 0.55, 0.0, 0.020), (0.95, -0.55, -0.40, 1.7, 0.016),
                (0.70, 1.05, 0.70, 3.1, 0.014))
        center = v3(0, 1.0, 0)
        for R, tilt, spd, ph, size in orbs:
            Rm = rot_z(tilt * 0.6) @ rot_x(tilt)
            oa = np.linspace(0, 2 * math.pi, 72)
            orbit = (np.stack([np.cos(oa) * R, np.zeros_like(oa), np.sin(oa) * R * 0.8], 1).astype(np.float32)
                     @ Rm.T) + center
            poly(orbit, (42, 37, 25), 1, True)
            a = t * spd + ph
            p3 = (v3(math.cos(a) * R, 0, math.sin(a) * R * 0.8) @ Rm.T) + center
            sx, sy, z = view.project(p3[None])
            if z[0] > 0.25:
                rr = max(2, int(size * view.f / z[0]))
                c0 = (int(sx[0]), int(sy[0]))
                cv2.circle(layer, c0, rr + 6, CYAN_DARK, -1, AA)
                cv2.circle(layer, c0, rr + 3, CYAN_DIM, -1, AA)
                cv2.circle(layer, c0, rr, CYAN, -1, AA)
                cv2.circle(layer, c0, max(rr // 2, 1), WHITE, -1, AA)

        # floating particles
        yy = (self.p_y + t * self.p_spd) % 1.0
        aa = self.p_ang + t * 0.15
        PP = np.stack([np.cos(aa) * self.p_rad, yy * 2.1, np.sin(aa) * self.p_rad], 1).astype(np.float32)
        sx, sy, z = view.project(PP)
        for i in range(self.pn):
            if z[i] < 0.25:
                continue
            br = (1.0 - yy[i]) * 0.9 + 0.1
            cv2.circle(layer, (int(sx[i]), int(sy[i])), 1 + (i % 3 == 0),
                       tuple(int(c * br) for c in CYAN), -1, AA)

        # bloom: blurred copy added back on top of the sharp lines
        sm = cv2.resize(layer, (max(w // 4, 1), max(h // 4, 1)), interpolation=cv2.INTER_AREA)
        sm = cv2.GaussianBlur(sm, (0, 0), 3)
        glow = cv2.resize(sm, (w, h), interpolation=cv2.INTER_LINEAR)
        layer = cv2.addWeighted(layer, 1.0, glow, 1.35, 0)
        canvas = cv2.add(canvas, layer)

        return canvas

    # ---- rendering
    def render(self, frame, t, fps):
        w, h = self.w, self.h
        view, rb = self.view, self.robot
        if self.cam_mode == CAM_FULL and frame is not None:
            canvas = video_background(frame, w, h, self.args.video_dim)
        else:
            canvas = studio_background(w, h)
        canvas = self.draw_stage(canvas, t)

        V, F, mat, pid, info = rb.build(t, self.explode)
        self.part_info = info
        self.stats = (len(F), len(V))
        L = np.zeros((h, w, 3), np.uint8)
        E = np.zeros((h, w, 3), np.uint8)
        view.draw_soup(L, E, V, F, mat, pid, self.selected, self.hover, 0.45 if rb.standby else 1.0)

        mask_full = L.max(axis=2) > 3
        rows = np.flatnonzero(mask_full.any(1))
        self.mini = None
        if len(rows):
            cols = np.flatnonzero(mask_full.any(0))
            m = 40
            r0, r1 = max(int(rows[0]) - m, 0), min(int(rows[-1]) + m, h)
            c0, c1 = max(int(cols[0]) - m, 0), min(int(cols[-1]) + m, w)
            Lr = L[r0:r1, c0:c1]
            Er = E[r0:r1, c0:c1]
            Lr[::3] = (Lr[::3] * 0.90).astype(np.uint8)          # hologram scanlines on the robot
            m8 = mask_full[r0:r1, c0:c1].astype(np.uint8) * 255
            hh, ww = m8.shape
            small = cv2.resize(m8, (max(ww // 4, 1), max(hh // 4, 1)), interpolation=cv2.INTER_AREA)
            glow = cv2.GaussianBlur(small, (0, 0), 4)
            glow = cv2.resize(glow, (ww, hh), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
            roi = canvas[r0:r1, c0:c1].astype(np.float32)
            out = roi + glow[..., None] * np.array(CYAN, np.float32) * 0.30
            # translucent glass body: the stage stays visible through the robot
            alpha = (cv2.GaussianBlur(m8, (3, 3), 0).astype(np.float32) / 255)[..., None] * 0.52
            out = out * (1 - alpha) + Lr.astype(np.float32) * alpha
            # Keep the wireframe crisp without washing out the shaded mesh.
            egl = cv2.GaussianBlur(Er, (0, 0), 3).astype(np.float32)
            out += Er.astype(np.float32) * 0.62 + egl * 0.58
            # restrained cyan silhouette outline
            edge = cv2.subtract(cv2.dilate(m8, np.ones((3, 3), np.uint8)), m8)
            edge = cv2.GaussianBlur(edge, (3, 3), 0).astype(np.float32) / 255
            out += edge[..., None] * np.array(CYAN, np.float32) * 0.48
            # scan band sweeping up the hologram
            by = (1.0 - (t * 0.32) % 1.3) * h - 0.15 * h
            band = np.exp(-((np.arange(r0, r1, dtype=np.float32) - by) / 12.0) ** 2)[:, None, None]
            out += (alpha + 0.25) * band * np.array(CYAN, np.float32) * 0.25
            canvas[r0:r1, c0:c1] = np.clip(out, 0, 255).astype(np.uint8)
            self.mini = cv2.addWeighted(Lr, 0.30, Er, 1.0, 0)

        # screen-space regions for picking
        self.regions = []
        for i, (c, r) in enumerate(info):
            sx, sy, z = view.project(np.asarray(c, np.float32)[None])
            self.regions.append((PART_NAMES[i], float(sx[0]), float(sy[0]),
                                 float(r * view.f / max(float(z[0]), 0.1)), float(z[0])))
        self.draw_ui(canvas, frame, t, fps)
        return canvas

    # ======================================================================= HOLOGRAM INTERFACE
    # ---- operator face / hand overlays (drawn on the webcam picture)
    def draw_face_overlay(self, image, x0, y0, bw, bh, small=True):
        if self.face is None:
            return
        for px, py in self.face["pts"]:
            cv2.circle(image, (x0 + int(px * bw), y0 + int(py * bh)), 1, (255, 255, 120), -1)
        bx0, by0, bx1, by1 = self.face["box"]
        brackets(image, x0 + int(bx0 * bw), y0 + int(by0 * bh), x0 + int(bx1 * bw), y0 + int(by1 * bh),
                 GOOD, 10 if small else 22, 2)

    def draw_hand_overlay(self, image, x0, y0, bw, bh, th):
        for hd in self.hands.values():
            pts = [(x0 + int(x * bw), y0 + int(y * bh)) for x, y in hd["lm"]]
            for a, b in HAND_CONN:
                cv2.line(image, pts[a], pts[b], CYAN, th, AA)
            for p in pts:
                cv2.circle(image, p, 2 if th == 1 else 3, WHITE, -1, AA)

    # ---- operator camera card (bottom right)
    def draw_cam_box(self, image, frame):
        h, w = image.shape[:2]
        px0, px1 = w - 290, w - 14
        x0, y0 = px0 + 8, h - 24 - BOX_H
        glass_panel(image, px0, y0 - 38, px1, h - 14, "OPERATOR CAM", CYAN, 14,
                    rule_end=px1 - 64)
        if frame is not None:
            vid = video_background(frame, BOX_W, BOX_H, 1.0).astype(np.float32)
        else:
            vid = np.full((BOX_H, BOX_W, 3), (22, 24, 8), np.float32)
        m = _rmask(BOX_W, BOX_H, 9).astype(np.float32)[..., None] / 255.0
        roi = image[y0:y0 + BOX_H, x0:x0 + BOX_W]
        roi[:] = (roi * (1 - m) + vid * m).astype(np.uint8)
        if frame is None:
            msg = "NO CAMERA"
            put_tracked(image, msg, (x0 + (BOX_W - tracked_width(msg, 0.34, 2)) // 2, y0 + BOX_H // 2 + 4),
                        0.34, INK3, 2)
        if self.show_hands:
            self.draw_face_overlay(image, x0, y0, BOX_W, BOX_H)
            self.draw_hand_overlay(image, x0, y0, BOX_W, BOX_H, 1)
        rrect_line(image, x0, y0, x0 + BOX_W, y0 + BOX_H, 9, HAIR2, 1)
        pulse = 0.55 + 0.45 * math.sin(time.time() * 4)
        cv2.circle(image, (px1 - 58, y0 - 18), 4, tuple(int(c * pulse) for c in RED), -1, AA)
        put_tracked(image, "LIVE", (px1 - 48, y0 - 14), 0.28, INK2, 2)
        pill(image, x0 + 8, y0 + BOX_H - 29, "FACE LOCK" if self.face is not None else "NO FACE",
             GOOD, self.face is not None, 20, 0.30)

    # ---- floating holographic cards beside the robot (BLUEPRINT + LIVE SCAN)
    def draw_deco(self, image, w, h, t):
        if self.view.zoom > 1.45 or w < 1100 or h < 640:
            return
        rb = self.robot
        # ---------- left: BLUEPRINT dial + joint bars
        x0, y0, x1, y1 = 268, 118, 424, 334
        glass_panel(image, x0, y0, x1, y1, "ROBOT DATA", CYAN, 14, alpha=0.72)
        cx, cy, r = (x0 + x1) // 2, y0 + 92, 38
        cv2.circle(image, (cx, cy), r + 8, CYAN_DARK, 1, AA)
        cv2.circle(image, (cx, cy), r, CYAN_DIM, 1, AA)
        a = int(t * 55) % 360
        cv2.ellipse(image, (cx, cy), (r, r), 0, a, a + 105, CYAN, 3, AA)
        cv2.ellipse(image, (cx, cy), (r - 9, r - 9), 0, 360 - a, 360 - a + 80, ORANGE, 2, AA)
        cv2.ellipse(image, (cx, cy), (r - 18, r - 18), 0, a * 2, a * 2 + 140, CYAN_DIM, 1, AA)
        for k in range(24):
            ang = math.radians(k * 15)
            l0, l1 = r + 3, r + (8 if k % 3 == 0 else 5)
            cv2.line(image, (int(cx + l0 * math.cos(ang)), int(cy + l0 * math.sin(ang))),
                     (int(cx + l1 * math.cos(ang)), int(cy + l1 * math.sin(ang))), HAIR2, 1, AA)
        n = str(len(PART_NAMES))
        txt(image, n, (cx - tw(n, 0.6, FONT_D) // 2, cy + 7), 0.6, INK1, 1, FONT_D, False)
        put_tracked(image, "PARTS", (cx - tracked_width("PARTS", 0.26, 1) // 2, cy + 20), 0.26, INK3, 1)
        vals = (("AL", rb.arm_raise["left"] / 150.0), ("AR", rb.arm_raise["right"] / 150.0),
                ("HY", abs(math.degrees(rb.head_yaw)) / 60.0), ("ZM", self.view.zoom / 3.0),
                ("EX", self.explode))
        base = y1 - 38
        bw_ = 18
        gap = (x1 - x0 - 32 - bw_ * len(vals)) / (len(vals) - 1)
        for i, (lab, v) in enumerate(vals):
            bx = int(x0 + 16 + i * (bw_ + gap))
            hgt = int(4 + 40 * float(np.clip(v, 0.0, 1.0)))
            tint_rrect(image, bx, base - 46, bx + bw_, base, 3, (62, 68, 40), 0.55)
            tint_rrect(image, bx, base - hgt, bx + bw_, base, 3, ORANGE if i == 4 else CYAN, 0.75)
            cv2.line(image, (bx, base - hgt), (bx + bw_, base - hgt), WHITE, 1, AA)
            put_tracked(image, lab, (bx + 2, base + 12), 0.25, INK3, 1)
        txt(image, f"TRIS {self.stats[0]:,}", (x0 + 16, y1 - 8), 0.30, INK3, 1, shadow=False)

        # ---------- right: live mesh preview
        sx0, sy0, sx1, sy1 = 836, 118, 976, 334
        glass_panel(image, sx0, sy0, sx1, sy1, "3D VIEW", ORANGE, 14, alpha=0.72)
        ax0, ay0, ax1, ay1 = sx0 + 12, sy0 + 34, sx1 - 12, sy1 - 34
        aw, ah = ax1 - ax0, ay1 - ay0
        tint_rrect(image, ax0, ay0, ax1, ay1, 6, (20, 22, 8), 0.60)
        if self.mini is not None:
            mh, mw = self.mini.shape[:2]
            sc = min(aw / mw, ah / mh)
            nw, nh = max(int(mw * sc), 2), max(int(mh * sc), 2)
            res = cv2.resize(self.mini, (nw, nh), interpolation=cv2.INTER_AREA)
            res = cv2.convertScaleAbs(res, alpha=1.8)
            ox, oy = ax0 + (aw - nw) // 2, ay0 + (ah - nh) // 2
            roi = image[oy:oy + nh, ox:ox + nw]
            roi[:] = cv2.add(roi, res)
        sy_ = ay0 + int(((t * 0.45) % 1.0) * ah)
        cv2.line(image, (ax0, sy_), (ax1, sy_), CYAN, 1, AA)
        brackets(image, ax0, ay0, ax1, ay1, ORANGE, 10, 1)
        put_tracked(image, "FACES", (sx0 + 16, sy1 - 12), 0.25, INK2, 1)
        mesh_count = f"{self.stats[0]:,}"
        txt(image, mesh_count, (sx1 - 16 - tw(mesh_count, 0.28), sy1 - 11), 0.28, GOOD, 1, shadow=False)

    # ---- exploded view labels
    def draw_part_labels(self, image, w, h):
        if self.explode <= 0.55:
            return
        for i, (name, sx, sy, r, _z) in enumerate(self.regions):
            if not (0 <= sx < w and 0 <= sy < h):
                continue
            sel = i == self.selected
            col = ORANGE if sel else CYAN
            cx, cy = int(sx), int(sy)
            right = sx >= w / 2
            lx, ly = cx + (26 if right else -26), cy - 14
            cv2.line(image, (cx, cy), (lx, ly), col if sel else CYAN_DIM, 1, AA)
            cv2.circle(image, (cx, cy), 3, col, -1, AA)
            label = name.title()
            lw = tw(label, 0.34) + 18
            bx = lx if right else lx - lw
            tint_rrect(image, bx, ly - 10, bx + lw, ly + 10, 10, (14, 16, 6), 0.80)
            rrect_line(image, bx, ly - 10, bx + lw, ly + 10, 10, col if sel else CYAN_DIM, 1)
            txt(image, label, (bx + 9, ly + 4), 0.34, INK1 if sel else INK2, 1, shadow=False)

    # ---- selected part reticle + label chip
    def draw_selection(self, image, t):
        if self.selected < 0 or not self.regions:
            return
        H, W = image.shape[:2]
        name, sx, sy, r, _z = self.regions[self.selected]
        cx, cy = int(sx), int(sy)
        rr = int(min(max(r, 16), 150) + 6)
        cv2.circle(image, (cx, cy), rr, ORANGE, 1, AA)
        pulse = (t * 1.2) % 1.0
        cv2.circle(image, (cx, cy), int(rr + pulse * 20),
                   tuple(int(c * (1 - pulse) * 0.8) for c in ORANGE), 1, AA)
        for k in range(4):
            a = math.degrees(t * 0.9) + k * 90
            cv2.ellipse(image, (cx, cy), (rr + 7, rr + 7), 0, a, a + 28, ORANGE, 2, AA)
        label = name
        lw = tracked_width(label, 0.30, 2) + 36
        bx = cx + rr + 22
        if bx + lw > W - 8:
            bx = cx - rr - 22 - lw
        by = int(np.clip(cy - 13, 80, H - 40))
        cv2.line(image, (cx + rr + 8 if bx > cx else cx - rr - 8, cy), (bx if bx > cx else bx + lw, by + 13),
                 ORANGE, 1, AA)
        tint_rrect(image, bx, by, bx + lw, by + 26, 13, (14, 16, 6), 0.85)
        tint_rrect(image, bx, by, bx + lw, by + 26, 13, ORANGE, 0.14)
        rrect_line(image, bx, by, bx + lw, by + 26, 13, ORANGE, 1)
        cv2.circle(image, (bx + 14, by + 13), 3, ORANGE, -1, AA)
        put_tracked(image, label, (bx + 25, by + 17), 0.30, INK1, 2)

    # ---- hand pointers
    def draw_pointers(self, image):
        hover_name = PART_NAMES[self.hover] if self.hover >= 0 else ""
        for hd in self.hands.values():
            px, py = int(hd["pointer"][0]), int(hd["pointer"][1])
            col = GOOD if hd["pinch"] else CYAN
            rad = 12 if hd["pinch"] else 17
            cv2.circle(image, (px, py), rad, col, 1, AA)
            cv2.circle(image, (px, py), rad + 6, tuple(int(c * 0.35) for c in col), 1, AA)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                cv2.line(image, (px + dx * (rad + 3), py + dy * (rad + 3)),
                         (px + dx * (rad + 9), py + dy * (rad + 9)), col, 1, AA)
            cv2.circle(image, (px, py), 4 if hd["pinch"] else 2, col, -1, AA)
        if hover_name and self.hands:
            hd = next(iter(self.hands.values()))
            hx, hy = int(hd["pointer"][0]) + 28, int(hd["pointer"][1]) - 24
            lw = tracked_width(hover_name, 0.28, 2) + 22
            tint_rrect(image, hx, hy, hx + lw, hy + 22, 11, (14, 16, 6), 0.85)
            rrect_line(image, hx, hy, hx + lw, hy + 22, 11, CYAN_DIM, 1)
            put_tracked(image, hover_name, (hx + 11, hy + 15), 0.28, INK1, 2)

    # ---- robot speech bubble (next to the head)
    def draw_speech_bubble(self, image, w, h, t):
        v = self.voice
        if v is None:
            return
        now = time.time()
        if v.thinking:
            bubble = "." * (1 + int(t * 3) % 3)
        elif v.reply and now - v.reply_t < len(v.reply) / CPS + 7.0:
            bubble = v.shown_reply()
        else:
            bubble = ""
        if not bubble:
            return
        lines = wrap_display_text(bubble, 300, 0.52) or [""]
        lines = lines[-7:]
        bw = int(math.ceil(max(_text_width(ln, 0.52) for ln in lines) + 40))
        bw = min(max(((bw + 15) // 16) * 16, 112), w - 32)
        bh = len(lines) * 23 + 52
        hx, hy = self.head_screen()
        hr = int(min(max(self.regions[0][3], 24), 90)) if self.regions else 40
        right = hx + hr + 48 + bw < w - 296
        bx0 = int(hx + hr + 48) if right else int(hx - hr - 48 - bw)
        bx0 = int(np.clip(bx0, 262, max(min(w - 300 - bw, w - bw - 8), 262)))
        by0 = int(np.clip(hy - bh // 2, 76, max(h - 300 - bh, 76)))
        glass_panel(image, bx0, by0, bx0 + bw, by0 + bh, None, CYAN, 16)
        cv2.circle(image, (bx0 + 19, by0 + 21), 3, CYAN, -1, AA)
        put_tracked(image, ROBOT_NAME.upper(), (bx0 + 30, by0 + 25), 0.30, INK2, 2)
        if v.talking():                                        # tiny animated equaliser
            for k in range(4):
                bh_ = int(3 + 6 * abs(math.sin(t * 9 + k * 1.3)))
                xx = bx0 + bw - 24 + k * 5
                cv2.line(image, (xx, by0 + 21 - bh_), (xx, by0 + 21 + bh_), ORANGE, 2, AA)
        for i, ln in enumerate(lines):
            txt(image, ln, (bx0 + 20, by0 + 54 + i * 23), 0.52, INK1, 1, shadow=False)
        if v.typing_active() and int(t * 3) % 2 == 0 and not v.thinking:
            tx = bx0 + 22 + int(_text_width(lines[-1], 0.52))
            cv2.line(image, (tx, by0 + 40 + (len(lines) - 1) * 23),
                     (tx, by0 + 58 + (len(lines) - 1) * 23), CYAN, 2)
        ax = bx0 if right else bx0 + bw
        tail_end = (int(hx + hr * (0.8 if right else -0.8)), int(hy))
        cv2.line(image, (ax, by0 + bh // 2), tail_end, CYAN_DIM, 1, AA)
        cv2.circle(image, tail_end, 7, tuple(int(c * 0.3) for c in CYAN), -1, AA)
        cv2.circle(image, tail_end, 3, CYAN, -1, AA)

    # ---- header bar
    def draw_header(self, image, w, h, t, fps):
        rb, v = self.robot, self.voice
        glass_panel(image, 14, 12, w - 14, 70, None, CYAN, 16)
        cx, cy = 47, 41
        a = int(t * 60) % 360
        cv2.circle(image, (cx, cy), 20, (62, 66, 34), 1, AA)
        cv2.ellipse(image, (cx, cy), (20, 20), 0, a, a + 110, CYAN, 2, AA)
        cv2.ellipse(image, (cx, cy), (14, 14), 0, 360 - a, 360 - a + 70, ORANGE, 1, AA)
        txt(image, "SA", (cx - 9, cy + 4), 0.4, INK1, 1, FONT_D, False)
        txt(image, ROBOT_NAME, (80, 38), 0.6, INK1, 1, FONT_D)
        put_tracked(image, f"ROBOTICS STUDIO  V{APP_VERSION}  {int(fps):02d}FPS", (81, 56), 0.27, INK3, 1)
        cv2.line(image, (300, 24), (300, 58), HAIR, 1, AA)

        if rb.standby:
            mode = "STANDBY"
        elif self.face is not None and self.hands:
            mode = "FOLLOWING YOU"
        elif self.face is not None:
            mode = "FACE TRACKING"
        elif self.hands:
            mode = "HAND TRACKING"
        else:
            mode = "DRAG TO EXPLORE" if self.args.no_camera else "LOOKING FOR YOU"

        mic_on = v is not None and v.mic_ok and not v.muted
        cursor = 316
        for label, active, color in (
                ("FACE", self.face is not None, CYAN),
                ("HANDS", bool(self.hands), CYAN),
                ("MIC", mic_on, ORANGE),
                ("ROBOT ACTIVE" if not rb.standby else "STANDBY", not rb.standby, GOOD)):
            cursor += pill(image, cursor, 29, label, color, active) + 8
        if cursor < w - 360:
            cv2.line(image, (cursor + 6, 24), (cursor + 6, 58), HAIR, 1, AA)
            put_tracked(image, "MODE", (cursor + 22, 36), 0.27, INK3, 2)
            txt(image, mode, (cursor + 22, 55), 0.40, CYAN if (self.face or self.hands) else INK2,
                1, shadow=False)
        logo_x = w - 310
        selected_x = max(cursor + 190, w - 480)
        selected_width = max(0, logo_x - selected_x - 14)
        if selected_width >= 48:
            sel = PART_NAMES[self.selected] if self.selected >= 0 else "FULL ROBOT"
            put_tracked(image, "VIEWING", (selected_x, 36), 0.27, INK3, 2)
            txt(image, fit_display_text(sel, selected_width, 0.40), (selected_x, 55), 0.40,
                ORANGE if self.selected >= 0 else INK1, 1, shadow=False)
        cv2.line(image, (w - 170, 24), (w - 170, 58), HAIR, 1, AA)
        logo_h, logo_w = self.logo_alpha.shape
        logo_y = (82 - logo_h) // 2
        roi = image[logo_y:logo_y + logo_h, logo_x:logo_x + logo_w].astype(np.float32)
        alpha = (self.logo_alpha.astype(np.float32) / 255.0)[..., None]
        logo = self.logo_color.astype(np.float32)
        image[logo_y:logo_y + logo_h, logo_x:logo_x + logo_w] = np.clip(
            roi * (1.0 - alpha) + logo * alpha, 0, 255
        ).astype(np.uint8)
        self.chat_button_bounds = chat_button(image, w - 156, 23)

    # ---- left column: head tracking + voice cheat-sheet
    def draw_left(self, image, w, h, t):
        x0, y0, x1, y1 = 14, 82, 250, 238
        live = self.face is not None
        glass_panel(image, x0, y0, x1, y1, "TRACKING", CYAN)
        yawd, pitd, rold = self.head_pose
        for i, (label, val, rng) in enumerate((("YAW", yawd, 60.0), ("PITCH", pitd, 35.0),
                                               ("ROLL", rold, 35.0))):
            yb = y0 + 52 + i * 26
            put_tracked(image, label, (x0 + 16, yb), 0.29, INK3, 2)
            s = f"{val:+.1f}"
            txt(image, s, (x1 - 16 - tw(s, 0.40), yb), 0.40, CYAN if live else INK3, 1, shadow=False)
            bipolar(image, x0 + 16, x1 - 16, yb + 12, val, rng, CYAN if live else INK3)
        if live:
            keycap(image, x0 + 16, y1 - 30, "C", "Recalibrate")
        else:
            prompt = "Mouse control enabled" if self.args.no_camera else "Move into camera view"
            txt(image, prompt, (x0 + 16, y1 - 14), 0.34, INK3, 1, shadow=False)

        ty0, ty1 = 250, h - 300
        glass_panel(image, x0, ty0, x1, ty1, "VOICE COMMANDS", CYAN)
        tips = ("Hello / who are you", "Show me the hand", "Explode / assemble",
                "Zoom in / zoom out", "Rotate / dance / wave", "Stand by / wake up",
                "Ask me anything")
        for i, tip in enumerate(tips):
            yb = ty0 + 54 + i * 17
            last = i == len(tips) - 1
            if last:
                cv2.drawMarker(image, (x0 + 20, yb - 4), CYAN, cv2.MARKER_DIAMOND, 7, 1, AA)
            else:
                cv2.circle(image, (x0 + 20, yb - 4), 2, HAIR2, -1, AA)
            txt(image, tip, (x0 + 32, yb), 0.37, CYAN if last else INK2, 1, shadow=False)

    # ---- right column: status rings + telemetry + AI guide
    def draw_right(self, image, w, h, t):
        rb = self.robot
        x0, y0, x1, y1 = w - 290, 82, w - 14, 366
        glass_panel(image, x0, y0, x1, y1, "SYSTEM OVERVIEW", CYAN)
        motion = float(np.clip(max(rb.arm_raise.values()) / 150.0 * 0.65 + self.explode * 0.35, 0, 1))
        tracking = self.face is not None or bool(self.hands)
        metrics = (
            ("ROBOT", "STANDBY" if rb.standby else "ACTIVE", 0.12 if rb.standby else 1.0, GOOD),
            ("SENSORS", "TRACKING" if tracking else "SEARCHING", 1.0 if tracking else 0.08, CYAN),
            ("MOTION", f"{motion * 100:.0f}%", motion, ORANGE if motion > 0.6 else CYAN),
        )
        gap = 8
        metric_w = (x1 - x0 - 32 - gap * 2) // 3
        metric_y = y0 + 38
        for i, (label, value, level, accent) in enumerate(metrics):
            mx = x0 + 12 + i * (metric_w + gap)
            tint_rrect(image, mx, metric_y, mx + metric_w, metric_y + 58, 8, (48, 39, 28), 0.34)
            cv2.circle(image, (mx + 11, metric_y + 14), 3, accent, -1, AA)
            put_tracked(image, label, (mx + 20, metric_y + 18), 0.23, INK3, 1)
            value_scale = 0.28 if len(value) > 7 else 0.34
            txt(image, value, (mx + 10, metric_y + 39), value_scale, INK1, 1, shadow=False)
            bar_x0, bar_x1 = mx + 10, mx + metric_w - 10
            tint_rrect(image, bar_x0, metric_y + 47, bar_x1, metric_y + 51, 2, HAIR, 0.9)
            if level > 0:
                tint_rrect(image, bar_x0, metric_y + 47,
                           bar_x0 + max(2, int((bar_x1 - bar_x0) * level)), metric_y + 51,
                           2, accent, 0.95)

        yy = metric_y + 76
        ex = put_tracked(image, "MOTION DATA", (x0 + 16, yy), 0.29, INK2, 2)
        cv2.line(image, (ex + 10, yy - 4), (x1 - 16, yy - 4), HAIR, 1, AA)
        wrapped = math.degrees((self.view.yaw + math.pi) % (2 * math.pi) - math.pi)
        rows = (("Head yaw", math.degrees(rb.head_yaw), "deg"), ("Head nod", math.degrees(rb.head_nod), "deg"),
                ("Arm raise L", rb.arm_raise["left"], "deg"), ("Arm raise R", rb.arm_raise["right"], "deg"),
                ("View yaw", wrapped, "deg"), ("Zoom", self.view.zoom, "x"),
                ("Explode", self.explode * 100, "%"))
        y = yy + 24
        for i, (label, val, unit_) in enumerate(rows):
            if i % 2 == 0:
                tint_rrect(image, x0 + 10, y - 13, x1 - 10, y + 5, 5, (53, 46, 37), 0.42)
            txt(image, label, (x0 + 18, y), 0.36, INK3, 1, shadow=False)
            s = f"{val:+.1f}"
            txt(image, unit_, (x1 - 18 - tw(unit_, 0.30), y), 0.30, INK3, 1, shadow=False)
            txt(image, s, (x1 - 26 - tw(unit_, 0.30) - tw(s, 0.38), y), 0.38, INK1, 1, shadow=False)
            y += 18

        # AI guide card
        ax0, ay0, ax1, ay1 = w - 290, 378, w - 14, 502
        glass_panel(image, ax0, ay0, ax1, ay1,
                    f"AI COPILOT  {self.guide.part}" if self.guide.part else "AI COPILOT", ORANGE)
        text = self.guide.text or self.guide.status
        all_lines = wrap_display_text(text, ax1 - ax0 - 34, 0.38)
        glines = all_lines[:4]
        if len(all_lines) > 4:
            glines[-1] = fit_display_text(glines[-1] + "...", ax1 - ax0 - 34, 0.38)
        for i, ln in enumerate(glines):
            txt(image, ln, (ax0 + 16, ay0 + 48 + i * 16), 0.38, INK1 if i == 0 else INK2, 1, shadow=False)
        cv2.line(image, (ax0 + 16, ay1 - 20), (ax1 - 16, ay1 - 20), HAIR, 1, AA)
        txt(image, fit_display_text(self.guide.status, ax1 - ax0 - 56, 0.30), (ax0 + 28, ay1 - 7),
            0.30, INK3, 1, shadow=False)
        cv2.circle(image, (ax0 + 19, ay1 - 10), 3,
                   GOOD if ("ready" in self.guide.status or "explanation" in self.guide.status) else INK3,
                   -1, AA)

    # ---- bottom left: conversation card
    def draw_voice(self, image, w, h, t):
        v = self.voice
        if v is None:
            return
        rb = self.robot
        vx0, vy0, vx1, vy1 = 14, h - 286, 590, h - 14
        st = "TYPING" if self.typing else v.state()
        stcol = {"SPEAKING": ORANGE, "THINKING": ORANGE, "HEARING YOU": GOOD, "LISTENING": CYAN,
                 "RECOGNIZING": GOOD, "TYPING": ORANGE}.get(st, INK3)
        active = st not in ("MUTED", "TYPE MODE")
        pw_ = pill_width(st, 0.32)
        glass_panel(image, vx0, vy0, vx1, vy1, "CONVERSATION", CYAN, 16, rule_end=vx1 - 16 - pw_ - 70)
        put_tracked(image, "ENGLISH", (vx1 - 16 - pw_ - 62, vy0 + 24), 0.27, INK3, 2)
        pill(image, vx1 - 16 - pw_, vy0 + 9, st, stcol, active, 22, 0.32)

        # live waveform
        talking = v.talking()
        lvl = v.level
        if talking:
            lvl = max(lvl * 0.3, 0.2 + 0.7 * abs(math.sin(t * 13)) * max(rb.mouth, 0.3))
        self.wave.append(lvl)
        nb = len(self.wave)
        inner = vx1 - vx0 - 32
        step_x = inner / nb
        mid = vy0 + 60
        wcol = ORANGE if talking else CYAN
        cv2.line(image, (vx0 + 16, mid), (vx1 - 16, mid), CYAN_DARK, 1, AA)
        for i, lv in enumerate(self.wave):
            hh = int(2 + lv * 17)
            xx = int(vx0 + 16 + i * step_x + step_x / 2)
            fade = 0.22 + 0.78 * (i / nb) ** 1.2
            col = tuple(int(c * fade) for c in wcol)
            cv2.line(image, (xx, mid - hh), (xx, mid + hh), col, 3, AA)

        # you
        by1, by2 = vy0 + 86, vy0 + 152
        bx0, bx1 = vx0 + 14, vx1 - 14
        maxw = bx1 - bx0 - 36
        if v.hearing:
            yt = "listening to you" + "." * (1 + int(t * 3) % 3)
        elif v.recognizing:
            yt = "recognizing" + "." * (1 + int(t * 3) % 3)
        elif self.typing:
            yt = "> " + self.typed + ("_" if int(t * 2) % 2 == 0 else " ")
        else:
            yt = v.heard or ("say hello to the robot" if v.mic_ok else "press Y and type to the robot")
        chat_bubble(image, bx0, by1, bx1, by2, "YOU TYPING" if self.typing else "YOU SAID", ORANGE,
                    strong=self.typing or v.hearing)
        ylines = wrap_display_text(yt, maxw, 0.48) or [""]
        if len(ylines) > 2:
            ylines = ylines[-2:] if self.typing else ylines[:2]
            if not self.typing:
                ylines[-1] = fit_display_text(ylines[-1] + "...", maxw, 0.48)
        ycol = INK1 if (v.heard and not v.hearing) or self.typing else INK3
        for i, ln in enumerate(ylines):
            txt(image, ln, (bx0 + 18, by1 + 40 + i * 19), 0.48, ycol, 1, shadow=False)

        # robot
        ry1, ry2 = vy0 + 160, vy0 + 244
        chat_bubble(image, bx0, ry1, bx1, ry2, ROBOT_NAME.upper(), CYAN, strong=talking or v.thinking)
        if v.thinking:
            rt = "thinking" + "." * (1 + int(t * 3) % 3)
        else:
            rt = v.shown_reply() if v.reply else "Ask me anything - I'm here to help."
        rlines = wrap_display_text(rt, maxw, 0.45) or [""]
        if len(rlines) > 3:
            rlines = rlines[:3]
            rlines[-1] = fit_display_text(rlines[-1] + "...", maxw, 0.45)
        for i, ln in enumerate(rlines):
            txt(image, ln, (bx0 + 18, ry1 + 40 + i * 19), 0.45, INK1 if v.reply else INK2, 1,
                shadow=False)

        # footer: mic status
        cv2.line(image, (vx0 + 16, vy1 - 26), (vx1 - 16, vy1 - 26), HAIR, 1, AA)
        mic_status = "ENTER send    ESC cancel    BACKSPACE delete" if self.typing else v.status
        cv2.circle(image, (vx0 + 21, vy1 - 12), 3, ORANGE if v.mic_ok and not v.muted else INK3, -1, AA)
        put_tracked(image, "MIC", (vx0 + 32, vy1 - 8), 0.27, INK2, 2)
        txt(image, fit_display_text(mic_status, vx1 - vx0 - 90, 0.33), (vx0 + 62, vy1 - 8), 0.33, INK3, 1,
            shadow=False)

    # ---- bottom centre: key-cap shortcut bar
    def draw_footer(self, image, w, h):
        items = (("DRAG", "Rotate"), ("SCROLL", "Zoom"), ("Y", "Chat"), ("E", "Explode"),
                 ("M", "Mute"), ("V", "Camera"), ("S", "Shot"))
        lo, hi, gap = 606, w - 300, 18
        chosen, total = [], 0
        for k, l in items:
            wd = keycap_width(k, l)
            add = wd + (gap if chosen else 0)
            if total + add > hi - lo - 28:
                break
            chosen.append((k, l, wd))
            total += add
        if not chosen:
            return
        px0 = int(lo + (hi - lo - total) / 2) - 14
        glass_panel(image, px0, h - 46, px0 + total + 28, h - 14, None, None, 16, alpha=0.88)
        x = px0 + 14
        for k, l, wd in chosen:
            keycap(image, x, h - 41, k, l)
            x += wd + gap

    # ---- standby banner
    def draw_standby(self, image, w, t):
        if not self.robot.standby:
            return
        msg = "STANDBY    OPEN A HAND OR SAY  WAKE UP"
        bw = tracked_width(msg, 0.32, 2) + 56
        x0 = (w - bw) // 2
        glass_panel(image, x0, 82, x0 + bw, 114, None, ORANGE, 16)
        pulse = 0.5 + 0.5 * math.sin(t * 3)
        cv2.circle(image, (x0 + 20, 98), 4, tuple(int(c * (0.4 + 0.6 * pulse)) for c in ORANGE), -1, AA)
        put_tracked(image, msg, (x0 + 34, 103), 0.32, INK1, 2)

    # ---- HUD root
    def draw_ui(self, image, frame, t, fps):
        h, w = image.shape[:2]
        # frosted-glass backdrop shared by every panel this frame
        small = cv2.resize(image, (max(w // 4, 1), max(h // 4, 1)), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 2.0)
        _UI["frost"] = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

        if self.cam_mode == CAM_FULL and self.show_hands:
            self.draw_face_overlay(image, 0, 0, w, h, small=False)
            self.draw_hand_overlay(image, 0, 0, w, h, 2)

        self.draw_deco(image, w, h, t)
        self.draw_part_labels(image, w, h)
        self.draw_selection(image, t)
        self.draw_pointers(image)
        self.draw_speech_bubble(image, w, h, t)
        self.draw_header(image, w, h, t, fps)
        self.draw_left(image, w, h, t)
        self.draw_right(image, w, h, t)
        if self.cam_mode == CAM_BOX:
            self.draw_cam_box(image, frame)
        self.draw_voice(image, w, h, t)
        self.draw_footer(image, w, h)
        self.draw_standby(image, w, t)


# --------------------------------------------------------------------------- main
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=f"{ROBOT_NAME} Robot 3D - Intelligent Robotics Studio")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--no-camera", action="store_true", help="no webcam / hand tracking (mouse only)")
    p.add_argument("--no-video", action="store_true", help="hide the webcam picture")
    p.add_argument("--no-face", action="store_true", help="disable head tracking")
    p.add_argument("--no-voice", action="store_true", help="disable the microphone (you can still type with Y)")
    p.add_argument("--no-speak", action="store_true", help="robot answers in text only (no spoken voice)")
    p.add_argument("--lang", default="en-IN",
                   help="English speech and reply locale (other languages use English)")
    p.add_argument("--list-languages", action="store_true", help="list available speech languages and exit")
    p.add_argument("--list-voices", action="store_true",
                   help="list installed macOS text-to-speech voices and exit")
    p.add_argument("--mic", type=int, default=None, help="microphone device index")
    p.add_argument("--full-video", action="store_true", help="webcam as full background (default: small box)")
    p.add_argument("--video-dim", type=float, default=0.55)
    p.add_argument("--no-rotate", action="store_true", help="(kept for compatibility, rotation is off by default)")
    p.add_argument("--rotate", action="store_true", help="rotate the camera image 180 degrees")
    p.add_argument("--model-complexity", type=int, choices=(0, 1), default=1)
    p.add_argument("--ollama-model", default=os.environ.get("HOLO_OLLAMA_MODEL", "qwen3:8b"))
    p.add_argument("--ollama-url", default=os.environ.get("HOLO_OLLAMA_URL", "http://localhost:11434"))
    p.add_argument("--offline", action="store_true", help="do not call Ollama")
    p.add_argument("--windowed", action="store_true", help="start in a resizable window instead of fullscreen")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    args = p.parse_args(argv)
    requested_language = args.lang or os.environ.get("SAM_ROBOT_LANG") or "en-IN"
    args.lang = resolve_speech_language(requested_language)
    if not requested_language.casefold().startswith(("en-", "english")):
        print(f"Only English is enabled; using English ({args.lang}).", file=sys.stderr)
    return args


def open_camera(index):
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    return cv2.VideoCapture(index, backend)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    if args.list_languages:
        print("Only English is enabled for speech recognition and chat replies.")
        print("Use --lang english or --lang en-IN.")
        return 0
    if args.list_voices:
        voices = Speaker._available_say_voices() if sys.platform == "darwin" else []
        if voices:
            print("Installed macOS speech voices:")
            for name, locale in voices:
                print(f"  {name:<24} {locale}")
        else:
            print("No macOS say voices found (or this is not macOS).")
        return 0
    args.video_dim = float(np.clip(args.video_dim, 0.0, 1.0))
    guide = OllamaGuide(args.ollama_url, args.ollama_model, enabled=not args.offline)
    speaker = Speaker(enabled=not args.no_speak, lang=args.lang)
    voice = Voice(guide, speaker, enabled=not args.no_voice, lang=args.lang, mic_index=args.mic)
    app = App(args, guide, voice)
    rotate180 = args.rotate

    cap = tracker = face_tracker = None
    if not args.no_camera:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise SystemExit("MediaPipe is required for hand tracking: python3 -m pip install mediapipe "
                             "(or run with --no-camera).") from exc
        cap = open_camera(args.camera)
        if not cap.isOpened():
            cap.release()
            raise SystemExit(f"Cannot open camera {args.camera}. Allow camera access for your terminal "
                             "or try --camera N.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        tracker = mp.solutions.hands.Hands(max_num_hands=2, model_complexity=args.model_complexity,
                                           min_detection_confidence=0.6, min_tracking_confidence=0.5)
        if not args.no_face:
            face_tracker = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1, refine_landmarks=False,
                min_detection_confidence=0.5, min_tracking_confidence=0.5)

    win = f"{ROBOT_NAME} Robot 3D v{APP_VERSION}  |  ROBOTICS STUDIO"
    display_size = [args.width, args.height]

    def refresh_display_size():
        _x, _y, display_w, display_h = cv2.getWindowImageRect(win)
        if display_w > 0 and display_h > 0:
            display_size[:] = [display_w, display_h]

    def display_frame(image):
        if image.shape[1] == display_size[0] and image.shape[0] == display_size[1]:
            return image
        shrinking = display_size[0] < image.shape[1] or display_size[1] < image.shape[0]
        interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
        return cv2.resize(image, tuple(display_size), interpolation=interpolation)

    def on_mouse(event, x, y, flags, _param):
        x = int(round(x * app.w / max(display_size[0], 1)))
        y = int(round(y * app.h / max(display_size[1], 1)))
        if event == cv2.EVENT_LBUTTONDOWN:
            app.m_down, app.m_last, app.m_moved = (x, y), (x, y), False
        elif event == cv2.EVENT_MOUSEMOVE and app.m_down is not None and (flags & cv2.EVENT_FLAG_LBUTTON):
            if math.hypot(x - app.m_down[0], y - app.m_down[1]) > 5:
                app.m_moved = True
            if app.m_moved:
                app.view.yaw -= (x - app.m_last[0]) * 0.010
                app.view.pitch = float(np.clip(app.view.pitch + (y - app.m_last[1]) * 0.006, -1.1, 1.1))
                app.view.auto = False
            app.m_last = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            if app.m_down is not None and not app.m_moved:
                if app.hit_chat_button(x, y):
                    app.want_chat = True
                else:
                    app.select(app.hit((x, y)))
            app.m_down = None
        elif event == cv2.EVENT_MOUSEWHEEL:
            delta = (flags >> 16) & 0xFFFF
            if delta >= 0x8000:
                delta -= 0x10000
            app.view.zoom_goal *= 1.12 if delta > 0 else 0.89

    def save_shot(img):
        name = f"shanib_aju_robot_{int(time.time())}.png"
        if not cv2.imwrite(name, img):
            raise OSError(f"could not save screenshot to {name}")
        print("saved", name)

    try:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        fullscreen = not args.windowed
        cv2.setWindowProperty(
            win, cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
        )
        cv2.waitKey(100)
        refresh_display_size()
        cv2.setMouseCallback(win, on_mouse)
        app.view.update(0.0, 0.0)
        app.render(None, 0.0, 60)                 # builds regions so the first selection works
        app.select(PID["HEAD"])
        t0 = prev = time.perf_counter()
        fps = 30.0
        print(f"{win}\nCamera tracking is {'OFF (--no-camera)' if cap is None else 'ON'}.\n"
              f"Speech and reply language: {speech_language_name(args.lang)} ({args.lang}).\n"
              f"Chat model: {guide.model} at {guide.url}.\n"
              f"{speaker.status}. English-only voice and chat. "
              "Press Y to type to the robot, M to mute the mic, U to toggle fullscreen.", flush=True)
        while True:
            now = time.perf_counter()
            dt = min(now - prev, 0.1)
            fps = 0.9 * fps + 0.1 / max(now - prev, 1e-3)
            prev = now
            t = now - t0
            frame, hands, face = None, {}, None
            if cap is not None:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"Camera {args.camera} stopped returning frames.")
                if rotate180:
                    frame = cv2.flip(frame, -1)
                frame = cv2.flip(frame, 1)
                if frame.shape[1] != args.width or frame.shape[0] != args.height:
                    frame = cv2.resize(frame, (args.width, args.height))
                # tracking runs on a smaller copy for speed (landmarks are normalised)
                small = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                hands = parse_hands(tracker.process(rgb), args.width, args.height)
                if face_tracker is not None:
                    face = parse_face(face_tracker.process(rgb))
            while not voice.inbox.empty():            # things you said / typed
                app.on_speech(voice.inbox.get())
            app.step(t, dt, hands, face)
            canvas = app.render(frame, t, fps)
            shown = display_frame(canvas)
            if app.want_shot:
                app.want_shot = False
                save_shot(shown)
            cv2.imshow(win, shown)
            refresh_display_size()
            k = cv2.waitKeyEx(1)
            k = k & 0xFF if k >= 0 else k
            if not app.typing and 0 <= k < 128 and chr(k).isalpha():
                k = ord(chr(k).lower())
            if k in (27, ord("q")):
                if app.typing:
                    app.handle_chat_key(k)
                else:
                    break
            elif app.typing:
                app.handle_chat_key(k)
            elif k == ord("y") or app.want_chat:
                app.want_chat = False
                app.open_chat()
            elif k == ord("m"):
                voice.muted = not voice.muted
            elif k == ord("x"):
                speaker.stop()
                voice.reply_t = 0.0
                voice.reply = ""
            elif k == ord("s"):
                save_shot(canvas)
            elif k in (ord("+"), ord("=")):
                app.view.zoom_goal *= 1.15
            elif k in (ord("-"), ord("_")):
                app.view.zoom_goal /= 1.15
            elif k == ord("0"):
                app.reset_view()
                app.explode_goal = 0.0
            elif k == ord("e"):
                app.toggle_explode()
            elif k == ord("f"):
                app.focus()
            elif k == ord("t"):
                app.view.auto = not app.view.auto
            elif k == ord("j"):
                app.view.yaw += 0.12
            elif k == ord("l"):
                app.view.yaw -= 0.12
            elif k == ord("k"):
                app.reset_view()
            elif k == ord("c"):
                app.calibrate()
            elif k == ord("n"):
                app.cycle(1)
            elif k == ord("b"):
                app.cycle(-1)
            elif k == ord("i") and app.selected >= 0:
                app.select(app.selected, force=True)
            elif k == ord("v"):
                app.cycle_cam()
            elif k == ord("h"):
                app.show_hands = not app.show_hands
            elif k == ord("r"):
                rotate180 = not rotate180
            elif k == ord("u"):
                fullscreen = not fullscreen
                cv2.setWindowProperty(
                    win, cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
                )
                if not fullscreen:
                    cv2.resizeWindow(win, args.width, args.height)
                cv2.waitKey(100)
                refresh_display_size()
    finally:
        speaker.stop()
        if tracker is not None:
            tracker.close()
        if face_tracker is not None:
            face_tracker.close()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())