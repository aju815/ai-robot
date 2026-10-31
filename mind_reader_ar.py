"""
Mind Reader AR - hand gestures trigger real-time AR effects
Stack: Python, OpenCV, MediaPipe, NumPy, ONNX Runtime

Pipeline:
    Webcam -> hand landmarks (+ face landmarks) -> gesture recognition
    -> action triggered -> AR effect drawn in real time

16 gestures (see the on-screen legend, key g):
    Peace (index+middle)         -> Goggles
    Thumbs up                    -> Dog
    Thumbs down                  -> Crying (tears + sad brows)
    Fist                         -> Cat
    Open palm (5 fingers)        -> Bow
    Thumb+index+pinky (ILY)      -> Makeup
    Index+pinky (rock)           -> Devil horns
    Thumb+pinky (call me)        -> Party hat
    OK sign (thumb+index circle) -> Mustache
    3 fingers (no pinky)         -> Crown
    4 fingers (no thumb)         -> Halo
    Thumb+index (gun)            -> Laser eyes
    Pinky only                   -> Sunglasses
    Heart (two hands)            -> Floating hearts
    Both palms open              -> Confetti
    Index pointing (hold)        -> Screenshot

MIND READER (heavy facial-expression AI, version 2):
    Press m, or point your index finger at your head (temple / forehead).
    An ENSEMBLE of 4 trained neural networks (EfficientNet-B0 x3 + EfficientNet-B2,
    trained on AffectNet / video emotion data, ~80 MB total) classifies your face into
    happy, surprised, sad, angry, fear, disgust or neutral with live probability bars.
    What makes it more accurate than a single model:
      * 4 models vote together (weighted average)
      * every model also sees a mirrored copy of your face (test-time augmentation)
      * your head tilt is removed first (the face crop is rotated upright)
      * a 2 second calibration learns YOUR resting face, so a naturally "serious"
        face is not read as angry or sad
      * the AI runs in a background thread, so the video stays smooth
      * frames where your hand covers your face are skipped (no wrong readings)
    Face-landmark rules add wink, sleepy and thinking.
    NOTE: it reads FACIAL EXPRESSION only. Nothing can read real thoughts, and no
    expression AI is 100% correct (the best ones reach about 60-70% on 8 emotions).
    The AI models download automatically the first time (needs internet).
    If they cannot load, the program falls back to simple landmark rules.

Keys:
    q quit        s screenshot     h hand landmarks on/off
    d debug (finger states)        g gesture legend on/off
    m mind reader on/off           c recalibrate mind reader (neutral face)
    n next effect (TEST MODE: shows every effect without any gesture)
    0 back to automatic gesture mode

Install:  pip install opencv-python "mediapipe==0.10.14" numpy onnxruntime
          (optional, NVIDIA GPU: pip install onnxruntime-gpu  instead of onnxruntime)
Run:      python mind_reader_ar.py

All effects are drawn with OpenCV shapes, so no image assets are needed.
"""

import base64
import json
import math
import os
import random
import shutil
import sys
import textwrap
import threading
import time
import urllib.request
from collections import Counter, deque
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np

# ------------------------------------------------------------------ settings
CAM_INDEX = 0
FRAME_W, FRAME_H = 1280, 720
PROC_W = 640               # frames are downscaled to this width for MediaPipe (faster)
VOTE_WINDOW = 5            # frames used to stabilise the gesture
VOTE_MIN = 3               # votes needed to accept a gesture
SCREENSHOT_HOLD = 10       # frames of pointing before a screenshot
SCREENSHOT_COOLDOWN = 3.0  # seconds
SCREENSHOT_DIR = "screenshots"

FINGER_BEND_ANGLE = 125    # degrees at the PIP joint; above = finger straight
THUMB_BEND_ANGLE = 130     # degrees at the thumb IP joint
THUMB_REACH = 0.40         # thumb tip distance from index knuckle / hand size (min)
PINCH_RATIO = 0.28         # thumb-tip to index-tip distance / hand size (OK sign)

# ------------------------------------------------------------------ gesture names
PEACE, THUMBS_UP, THUMBS_DOWN, FIST = "PEACE", "THUMBS_UP", "THUMBS_DOWN", "FIST"
OPEN_PALM, ILY, ROCK, CALL = "OPEN_PALM", "ILY", "ROCK", "CALL"
OK, THREE, FOUR, GUN = "OK", "THREE", "FOUR", "GUN"
PINKY, HEART, CONFETTI, POINTING, NONE = "PINKY", "HEART", "CONFETTI", "POINTING", "NONE"

GESTURE_INFO = [
    (PEACE, "Peace (index+middle)", "Goggles"),
    (THUMBS_UP, "Thumbs up", "Dog"),
    (THUMBS_DOWN, "Thumbs down", "Crying"),
    (FIST, "Fist", "Cat"),
    (OPEN_PALM, "Open palm (5)", "Bow"),
    (ILY, "Thumb+index+pinky", "Makeup"),
    (ROCK, "Index+pinky (rock)", "Devil horns"),
    (CALL, "Thumb+pinky (call)", "Party hat"),
    (OK, "OK sign", "Mustache"),
    (THREE, "3 fingers (no pinky)", "Crown"),
    (FOUR, "4 fingers (no thumb)", "Halo"),
    (GUN, "Thumb+index (gun)", "Laser eyes"),
    (PINKY, "Pinky only", "Sunglasses"),
    (HEART, "Heart (2 hands)", "Hearts"),
    (CONFETTI, "Both palms open", "Confetti"),
    (POINTING, "Index only (hold)", "Screenshot"),
]
EFFECT_NAMES = {g: e for g, _, e in GESTURE_INFO}
EFFECT_NAMES[NONE] = "-"
TEST_ORDER = [g for g, _, _ in GESTURE_INFO]

LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269,
        267, 0, 37, 39, 40, 185]
R_UPPER_LID = [33, 246, 161, 160, 159, 158, 157, 173, 133]
L_UPPER_LID = [362, 398, 384, 385, 386, 387, 388, 466, 263]


# ------------------------------------------------------------------ drawing helpers
def ip(p):
    return int(p[0]), int(p[1])


def poly(img, pts, color):
    cv2.fillPoly(img, [np.array([ip(p) for p in pts], np.int32)], color)


def outline(img, pts, color, thick=2, closed=True):
    cv2.polylines(img, [np.array([ip(p) for p in pts], np.int32)], closed,
                  color, thick, cv2.LINE_AA)


def alpha_draw(frame, draw_fn, alpha):
    """Draw something semi-transparent on the frame."""
    overlay = frame.copy()
    draw_fn(overlay)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def heart_points(cx, cy, size, n=40):
    t = np.linspace(0, 2 * math.pi, n)
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    s = size / 16.0
    return [(cx + xi * s, cy - yi * s) for xi, yi in zip(x, y)]


def star_points(cx, cy, r):
    k = 0.28
    return [(cx, cy - r), (cx + r * k, cy - r * k), (cx + r, cy), (cx + r * k, cy + r * k),
            (cx, cy + r), (cx - r * k, cy + r * k), (cx - r, cy), (cx - r * k, cy - r * k)]


# ------------------------------------------------------------------ gesture recognition
def angle(a, b, c):
    v1, v2 = a - b, c - b
    d = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6
    return math.degrees(math.acos(float(np.clip(np.dot(v1, v2) / d, -1, 1))))


def hand_size(L):
    return float(np.linalg.norm(L[0] - L[9])) + 1e-6


FINGER_IDX = {"index": (5, 6, 8), "middle": (9, 10, 12),
              "ring": (13, 14, 16), "pinky": (17, 18, 20)}


def finger_states(L):
    """L = (21, 2) array of landmark PIXEL coordinates.
    Rotation-invariant: a finger is extended when its PIP joint is straight and
    the tip is farther from the wrist than the PIP joint (works at any hand angle).
    The thumb is tested by its joint angle and by how far it reaches from the knuckle."""
    wrist = L[0]
    st = {}
    for name, (mcp, pip, tip) in FINGER_IDX.items():
        st[name] = bool(angle(L[mcp], L[pip], L[tip]) > FINGER_BEND_ANGLE and
                        np.linalg.norm(L[tip] - wrist) > np.linalg.norm(L[pip] - wrist))
    # thumb: straight, reaching away from the index knuckle, and its tip farther from
    # the pinky knuckle than its own middle joint (a folded thumb crosses the palm)
    st["thumb"] = bool(angle(L[2], L[3], L[4]) > THUMB_BEND_ANGLE and
                       np.linalg.norm(L[4] - L[5]) > THUMB_REACH * hand_size(L) and
                       np.linalg.norm(L[4] - L[17]) > 1.05 * np.linalg.norm(L[3] - L[17]))
    return st


def classify(L):
    """Return (gesture, finger_states) for one hand."""
    s = finger_states(L)
    t, i, m, r, p = s["thumb"], s["index"], s["middle"], s["ring"], s["pinky"]
    hs = hand_size(L)
    pinch = np.linalg.norm(L[4] - L[8]) < PINCH_RATIO * hs

    if pinch and m and r and p:
        return OK, s
    if t and i and m and r and p:
        return OPEN_PALM, s
    if i and m and r and p and not t:
        return FOUR, s
    if i and m and r and not p:
        return THREE, s
    if i and m and not r and not p:
        return PEACE, s
    if t and i and p and not m and not r:
        return ILY, s
    if i and p and not (t or m or r):
        return ROCK, s
    if t and p and not (i or m or r):
        return CALL, s
    if t and i and not (m or r or p):
        return GUN, s
    if i and not (t or m or r or p):
        return POINTING, s
    if p and not (t or i or m or r):
        return PINKY, s
    if t and not (i or m or r or p):
        v = L[4] - L[2]
        n = np.linalg.norm(v) + 1e-6
        if v[1] < -0.6 * n:
            return THUMBS_UP, s
        if v[1] > 0.6 * n:
            return THUMBS_DOWN, s
        return NONE, s
    if not (t or i or m or r or p):
        return FIST, s
    return NONE, s


def is_heart(hands):
    """Two hands: thumb tips touching and index tips touching."""
    if len(hands) < 2:
        return False
    a, b = hands[0], hands[1]
    thr = 0.8 * max(hand_size(a), hand_size(b))
    return (np.linalg.norm(a[4] - b[4]) < thr and
            np.linalg.norm(a[8] - b[8]) < thr * 1.4)


class GestureStabilizer:
    """Majority vote over the last few frames to remove flicker."""

    def __init__(self):
        self.buf = deque(maxlen=VOTE_WINDOW)
        self.current = NONE

    def update(self, g):
        self.buf.append(g)
        name, count = Counter(self.buf).most_common(1)[0]
        if count >= VOTE_MIN:
            self.current = name
        return self.current


# ------------------------------------------------------------------ face frame
class FaceFrame:
    """Face landmarks plus a head-aligned coordinate system, so effects follow
    head size and tilt. Offsets are measured in face widths."""

    def __init__(self, lm, w, h):
        self.lm, self.w, self.h = lm, w, h
        a, b = self.pt(234), self.pt(454)
        self.fw = float(np.linalg.norm(b - a)) + 1e-6
        th = math.atan2(b[1] - a[1], b[0] - a[0])
        self.deg = math.degrees(th)
        self.r = np.array([math.cos(th), math.sin(th)])    # head "right"
        self.u = np.array([math.sin(th), -math.cos(th)])   # head "up"

    def pt(self, i):
        return np.array([self.lm[i].x * self.w, self.lm[i].y * self.h])

    def at(self, base, x, y):
        return base + (x * self.r + y * self.u) * self.fw

    def eyes(self):
        le = (self.pt(33) + self.pt(133)) / 2
        re = (self.pt(362) + self.pt(263)) / 2
        return le, re


# ------------------------------------------------------------------ AR effects (face)
def fx_goggles(frame, f):
    le, re = f.eyes()
    rad = int(0.115 * f.fw)
    cv2.line(frame, ip(f.pt(234)), ip(le - f.r * rad), (40, 40, 40), 9)
    cv2.line(frame, ip(f.pt(454)), ip(re + f.r * rad), (40, 40, 40), 9)
    cv2.line(frame, ip(le + f.r * rad), ip(re - f.r * rad), (60, 60, 60), 7)
    for c in (le, re):
        alpha_draw(frame, lambda o, c=c: cv2.circle(o, ip(c), rad, (255, 160, 40), -1), 0.45)
        cv2.circle(frame, ip(c), rad, (30, 30, 30), 7)
        cv2.circle(frame, ip(c), rad + 4, (0, 200, 255), 2)
        cv2.ellipse(frame, ip(c - f.r * rad * 0.3 + f.u * rad * 0.35),
                    (int(rad * 0.25), int(rad * 0.1)), f.deg - 35, 0, 360,
                    (255, 255, 255), -1)


def fx_sunglasses(frame, f):
    le, re = f.eyes()
    for c in (le, re):
        pts = [f.at(c, -0.10, 0.06), f.at(c, 0.10, 0.06),
               f.at(c, 0.09, -0.075), f.at(c, -0.09, -0.075)]
        poly(frame, pts, (20, 20, 20))
        outline(frame, pts, (0, 0, 0), 4)
        cv2.line(frame, ip(f.at(c, -0.06, 0.035)), ip(f.at(c, -0.01, 0.035)),
                 (200, 200, 200), 2, cv2.LINE_AA)
    cv2.line(frame, ip(f.at(le, 0.10, 0.04)), ip(f.at(re, -0.10, 0.04)), (0, 0, 0), 5)
    cv2.line(frame, ip(f.at(le, -0.10, 0.04)), ip(f.pt(234)), (0, 0, 0), 5)
    cv2.line(frame, ip(f.at(re, 0.10, 0.04)), ip(f.pt(454)), (0, 0, 0), 5)


def fx_bow(frame, f):
    c = f.at(f.pt(10), 0.0, 0.10)
    pink, edge = (200, 120, 255), (130, 40, 190)
    for sx in (-1, 1):
        wing = [c, f.at(c, sx * 0.20, 0.10), f.at(c, sx * 0.22, 0.0),
                f.at(c, sx * 0.20, -0.10)]
        poly(frame, wing, pink)
        outline(frame, wing, edge, 2)
    cv2.circle(frame, ip(c), int(0.04 * f.fw), (160, 70, 220), -1)
    cv2.circle(frame, ip(c), int(0.04 * f.fw), edge, 2)


def fx_cat(frame, f):
    top = f.pt(10)
    for sx in (-1, 1):
        base = f.at(top, sx * 0.24, -0.04)
        outer = [f.at(base, -0.11, 0), f.at(base, 0.11, 0), f.at(base, sx * 0.05, 0.27)]
        inner = [f.at(base, -0.06, 0.01), f.at(base, 0.06, 0.01), f.at(base, sx * 0.04, 0.19)]
        poly(frame, outer, (70, 70, 80))
        poly(frame, inner, (180, 150, 255))
    n = f.pt(1)
    poly(frame, [f.at(n, -0.03, 0.015), f.at(n, 0.03, 0.015), f.at(n, 0, -0.02)],
         (150, 100, 255))
    for sx in (-1, 1):
        start = f.at(n, sx * 0.09, -0.03)
        for dy in (-0.05, 0.0, 0.05):
            end = f.at(start, sx * 0.28, dy)
            cv2.line(frame, ip(start), ip(end), (255, 255, 255), 2, cv2.LINE_AA)


def fx_dog(frame, f):
    top = f.pt(10)
    for sx in (-1, 1):
        c = f.at(top, sx * 0.30, -0.10)
        axes = (int(0.09 * f.fw), int(0.18 * f.fw))
        cv2.ellipse(frame, ip(c), axes, f.deg - sx * 15, 0, 360, (30, 70, 120), -1)
        cv2.ellipse(frame, ip(c), axes, f.deg - sx * 15, 0, 360, (15, 40, 80), 3)
    n = f.pt(1)
    cv2.ellipse(frame, ip(n), (int(0.055 * f.fw), int(0.04 * f.fw)), f.deg, 0, 360,
                (20, 20, 20), -1)
    cv2.circle(frame, ip(f.at(n, -0.02, 0.012)), max(int(0.012 * f.fw), 2),
               (255, 255, 255), -1)
    if np.linalg.norm(f.pt(13) - f.pt(14)) > 0.04 * f.fw:     # mouth open -> tongue
        t = f.at(f.pt(17), 0, -0.03)
        cv2.ellipse(frame, ip(t), (int(0.04 * f.fw), int(0.06 * f.fw)), f.deg, 0, 360,
                    (140, 110, 255), -1)
        cv2.line(frame, ip(t), ip(f.at(t, 0, -0.05)), (90, 70, 200), 2)


def fx_makeup(frame, f):
    lips = np.array([ip(f.pt(i)) for i in LIPS], np.int32)
    alpha_draw(frame, lambda o: cv2.fillPoly(o, [lips], (50, 20, 210)), 0.70)
    cv2.polylines(frame, [lips], True, (30, 10, 150), 2, cv2.LINE_AA)
    cv2.ellipse(frame, ip(f.at(f.pt(17), -0.02, 0.03)),
                (int(0.03 * f.fw), max(int(0.008 * f.fw), 1)), f.deg, 0, 360,
                (230, 210, 255), -1)                               # lip gloss
    for i in (205, 425):                                           # blush
        alpha_draw(frame, lambda o, i=i: cv2.circle(o, ip(f.pt(i)), int(0.11 * f.fw),
                                                    (160, 110, 255), -1), 0.40)
    for lid in (R_UPPER_LID, L_UPPER_LID):                         # eyeshadow + liner
        pts = [f.pt(i) for i in lid]
        shadow = [f.at(p, 0, 0.04) for p in pts]
        alpha_draw(frame, lambda o, s=shadow, p=pts: cv2.fillPoly(
            o, [np.array([ip(q) for q in s + p[::-1]], np.int32)], (200, 70, 160)), 0.55)
        outline(frame, pts, (20, 20, 20), 3, closed=False)
    t = time.time()                                                # sparkles
    for k, i in enumerate((205, 425, 10)):
        r = (0.02 + 0.015 * abs(math.sin(t * 3 + k))) * f.fw
        c = f.at(f.pt(i), 0.06 if k < 2 else 0.1, 0.07)
        poly(frame, star_points(c[0], c[1], r), (255, 255, 255))


def fx_tears(frame, f):
    le, re = f.eyes()
    t = time.time()
    brow = (30, 40, 60)
    cv2.line(frame, ip(f.at(le, -0.09, 0.09)), ip(f.at(le, 0.07, 0.15)), brow, 5, cv2.LINE_AA)
    cv2.line(frame, ip(f.at(re, -0.07, 0.15)), ip(f.at(re, 0.09, 0.09)), brow, 5, cv2.LINE_AA)
    drops = []
    for sx, c in ((-1, le), (1, re)):
        for k in range(2):
            ph = (t * 0.7 + k * 0.5 + (0.25 if sx > 0 else 0)) % 1.0
            pos = f.at(c, sx * 0.03, -0.07 - ph * 0.30)
            drops.append((pos, 1.0 - ph))

    def draw(o):
        for pos, life in drops:
            rd = int(0.022 * f.fw)
            cv2.circle(o, ip(pos), rd, (255, 190, 80), -1)
            poly(o, [f.at(pos, -0.02, 0.0), f.at(pos, 0.02, 0.0), f.at(pos, 0, 0.06)],
                 (255, 190, 80))
    alpha_draw(frame, draw, 0.8)


def fx_horns(frame, f):
    top = f.pt(10)
    for sx in (-1, 1):
        base = f.at(top, sx * 0.17, -0.03)
        horn = [f.at(base, -0.05, 0), f.at(base, 0.05, 0),
                f.at(base, sx * 0.06, 0.13), f.at(base, sx * 0.11, 0.25)]
        poly(frame, horn, (0, 0, 210))
        outline(frame, horn, (0, 0, 110), 2)


def fx_party_hat(frame, f):
    base = f.at(f.pt(10), 0.0, 0.03)
    bl, br, tip = f.at(base, -0.17, 0), f.at(base, 0.17, 0), f.at(base, 0.03, 0.55)
    poly(frame, [bl, br, tip], (200, 80, 230))
    for k in (0.25, 0.5, 0.75):
        a = bl + (tip - bl) * k
        b = br + (tip - br) * k
        cv2.line(frame, ip(a), ip(b), (0, 230, 255), 4, cv2.LINE_AA)
    outline(frame, [bl, br, tip], (120, 30, 150), 2)
    cv2.circle(frame, ip(tip), int(0.04 * f.fw), (0, 230, 255), -1)


def fx_crown(frame, f):
    base = f.at(f.pt(10), 0.0, 0.0)
    shape = [(-0.17, 0), (-0.17, 0.17), (-0.085, 0.08), (0, 0.22), (0.085, 0.08),
             (0.17, 0.17), (0.17, 0)]
    pts = [f.at(base, x, y) for x, y in shape]
    poly(frame, pts, (0, 205, 255))
    outline(frame, pts, (0, 120, 190), 3)
    cv2.line(frame, ip(f.at(base, -0.17, 0.035)), ip(f.at(base, 0.17, 0.035)),
             (0, 120, 190), 2)
    for (x, y), col in zip([(-0.17, 0.17), (0, 0.22), (0.17, 0.17)],
                           [(0, 0, 230), (230, 80, 0), (0, 0, 230)]):
        cv2.circle(frame, ip(f.at(base, x, y + 0.02)), int(0.022 * f.fw), col, -1)


def fx_halo(frame, f):
    c = f.at(f.pt(10), 0.0, 0.24)
    axes = (int(0.17 * f.fw), int(0.045 * f.fw))
    alpha_draw(frame, lambda o: cv2.ellipse(o, ip(c), axes, f.deg, 0, 360,
                                            (0, 220, 255), 16), 0.35)
    cv2.ellipse(frame, ip(c), axes, f.deg, 0, 360, (0, 230, 255), 6, cv2.LINE_AA)


def fx_laser(frame, f):
    le, re = f.eyes()
    length = max(f.w, f.h) * 1.5
    flick = random.uniform(0.8, 1.2)
    beams = []
    for c, sx in ((le, -1), (re, 1)):
        d = sx * f.r * 0.95 - f.u * 0.25
        d = d / np.linalg.norm(d)
        beams.append((c, c + d * length))

    def glow(o):
        for a, b in beams:
            cv2.line(o, ip(a), ip(b), (0, 0, 255), int(28 * flick), cv2.LINE_AA)
    alpha_draw(frame, glow, 0.35)
    for a, b in beams:
        cv2.line(frame, ip(a), ip(b), (60, 60, 255), int(12 * flick), cv2.LINE_AA)
        cv2.line(frame, ip(a), ip(b), (255, 255, 255), 4, cv2.LINE_AA)
        cv2.circle(frame, ip(a), int(0.035 * f.fw * flick), (255, 255, 255), -1)


def fx_mustache(frame, f):
    base = f.at(f.pt(164), 0, -0.005)
    thick = max(int(0.04 * f.fw), 3)
    for sx in (-1, 1):
        path = [f.at(base, sx * x, y) for x, y in
                [(0.0, 0.0), (0.05, -0.025), (0.10, -0.022), (0.15, 0.0), (0.175, 0.035)]]
        outline(frame, path, (25, 25, 25), thick, closed=False)
        outline(frame, [f.at(p, 0, 0.012) for p in path], (60, 60, 60), max(thick // 3, 1),
                closed=False)


FACE_FX = {
    PEACE: fx_goggles, THUMBS_UP: fx_dog, THUMBS_DOWN: fx_tears, FIST: fx_cat,
    OPEN_PALM: fx_bow, ILY: fx_makeup, ROCK: fx_horns, CALL: fx_party_hat,
    OK: fx_mustache, THREE: fx_crown, FOUR: fx_halo, GUN: fx_laser,
    PINKY: fx_sunglasses,
}


# ------------------------------------------------------------------ particle effects
class HeartParticles:
    def __init__(self):
        self.items = []

    def spawn(self, x, y):
        self.items.append({
            "x": x + random.uniform(-40, 40), "y": y,
            "vx": random.uniform(-1.2, 1.2), "vy": random.uniform(-4.5, -2.0),
            "size": random.uniform(14, 36), "life": 1.0,
            "color": random.choice([(120, 80, 255), (180, 120, 255), (90, 60, 230)]),
        })

    def update_and_draw(self, frame):
        alive = []
        for p in self.items:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["life"] -= 0.012
            if p["life"] > 0:
                pts = heart_points(p["x"], p["y"], p["size"])
                alpha_draw(frame, lambda o, pts=pts, c=p["color"]: poly(o, pts, c),
                           max(p["life"], 0.1))
                alive.append(p)
        self.items = alive


class Confetti:
    COLORS = [(0, 0, 255), (0, 200, 255), (0, 220, 0), (255, 120, 0),
              (255, 0, 200), (255, 255, 0)]

    def __init__(self):
        self.items = []

    def spawn(self, w, n=6):
        for _ in range(n):
            self.items.append({
                "x": random.uniform(0, w), "y": -10.0,
                "vx": random.uniform(-1.5, 1.5), "vy": random.uniform(3, 8),
                "a": random.uniform(0, 6.28), "va": random.uniform(-0.3, 0.3),
                "s": random.uniform(6, 12), "c": random.choice(self.COLORS),
            })

    def update_and_draw(self, frame):
        h = frame.shape[0]
        alive = []
        for p in self.items:
            p["x"] += p["vx"] + math.sin(p["y"] * 0.03)
            p["y"] += p["vy"]
            p["a"] += p["va"]
            if p["y"] < h + 20:
                ca, sa, s = math.cos(p["a"]), math.sin(p["a"]), p["s"]
                pts = [(p["x"] + dx * ca - dy * sa, p["y"] + dx * sa + dy * ca)
                       for dx, dy in [(-s, -s / 2), (s, -s / 2), (s, s / 2), (-s, s / 2)]]
                poly(frame, pts, p["c"])
                alive.append(p)
        self.items = alive


# ------------------------------------------------------------------ mind reader (heavy AI ensemble)
FACE_KEYS = ["HAPPY", "SURPRISED", "SAD", "ANGRY", "FEAR", "DISGUST", "NEUTRAL"]   # from the AI models
EXTRA_KEYS = ["SLEEPY", "WINK", "THINKING"]                                         # from landmarks
MIND_EMOTIONS = FACE_KEYS + EXTRA_KEYS

MODEL_BASE = ("https://raw.githubusercontent.com/HSE-asavchenko/face-emotion-recognition/"
              "main/models/affectnet_emotions/onnx/")
MODEL_DIR = os.path.join(os.path.expanduser("~"), ".mind_reader")
# (file name, vote weight). All output 8 AffectNet classes in the same order.
# EfficientNet-B2 uses 260x260 input, the B0 models use 224x224 (read from the model itself).
MODEL_SPECS = [
    ("enet_b0_8_best_vgaf.onnx", 1.0),
    ("enet_b0_8_best_afew.onnx", 1.0),
    ("enet_b0_8_va_mtl.onnx", 1.0),      # 10 outputs: 8 emotions + valence + arousal (we use the 8)
    ("enet_b2_8.onnx", 0.6),
]
CNN_CLASSES = ["Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]
CNN_TO_MIND = {"Anger": "ANGRY", "Contempt": "DISGUST", "Disgust": "DISGUST", "Fear": "FEAR",
               "Happiness": "HAPPY", "Neutral": "NEUTRAL", "Sadness": "SAD",
               "Surprise": "SURPRISED"}
IMG_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMG_STD = np.array([0.229, 0.224, 0.225], np.float32)
CROP_SIZE = 260            # face crop is made at the biggest model size, then resized per model
CROP_MARGIN = 0.08         # extra border around the face (fraction of face size)
SOFTMAX_TEMP = 1.5         # >1 makes over-confident networks a bit less extreme
AI_ALPHA = 0.45            # smoothing of AI results over time (higher = faster, jumpier)
AI_MIN_CONF = 0.40         # an emotion must reach this probability, otherwise NEUTRAL
BIAS_STRENGTH = 0.8        # how much of your resting-face bias is removed after calibration
BIAS_CAP = 0.6             # never remove more than this per emotion
# ---- BIG model (multi-GB vision-language AI running locally through Ollama, optional)
BIG_MODEL = os.environ.get("MIND_BIG_MODEL", "qwen2.5vl:7b")   # ~6 GB. Lighter: "qwen2.5vl:3b" or "gemma3:4b" (~3 GB)
BIG_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
BIG_WEIGHT = 0.6           # share of the big model in the final answer (0 = ignore it)
BIG_THOUGHT_TIME = 12.0    # seconds a thought written by the big model stays on screen
BIG_MAX_AGE = 25.0         # seconds a big-model answer stays valid
BIG_CROP = 448             # face image size sent to the big model
CAL_FRAMES = 35            # valid frames needed for calibration
CAL_AI = 3                 # AI results needed for calibration
CAL_TIMEOUT = 15.0         # seconds, calibration gives up and uses what it has
MIND_TIME = 8.0            # seconds the mind reader stays on after pointing at your head
HEAD_TOUCH_RATIO = 0.5     # fingertip-to-head distance / face width that counts as "pointing at head"
HAND_COVER_POINTS = 3      # hand landmarks inside the face box = face is covered

THOUGHTS = {
    "HAPPY": ["Today is a good day!", "I just remembered something funny...",
              "Life is great!", "Best mood ever!"],
    "LAUGHING": ["HAHAHA! That is hilarious!", "I cannot stop laughing!",
                 "Okay okay, that was funny."],
    "FEAR": ["Oh no... what was that?!", "Did I forget the deadline?!",
             "Was that a noise?!"],
    "DISGUST": ["Ew, what is that smell?", "I really do not like this.",
                "That food looks suspicious..."],
    "SURPRISED": ["Wait... WHAT?!", "Did I leave the stove on?!", "No way!",
                  "I did not see that coming!"],
    "SAD": ["I need a hug...", "Why is Monday so long?", "Maybe ice cream will help.",
            "Today was a lot."],
    "ANGRY": ["Who ate my food?!", "This Wi-Fi is SO slow!", "I am trying to stay calm...",
              "Why is my code not working?!"],
    "SLEEPY": ["Five more minutes...", "I need coffee.", "Zzz... is it bedtime yet?",
               "My bed is calling me."],
    "WINK": ["Shh, it is our secret!", "Just kidding!", "Nice one, right?"],
    "THINKING": ["Hmm, what should I eat?", "Let me think about this...",
                 "Is this code correct?", "What if I try another idea?"],
    "NEUTRAL": ["Just vibing.", "What is for dinner?", "Thinking about nothing...",
                "Calm and focused."],
}


def clip01(x):
    return float(np.clip(x, 0.0, 1.0))


def download_model(name):
    """Download one model file into MODEL_DIR (skipped if it is already there)."""
    path = os.path.join(MODEL_DIR, name)
    if os.path.isfile(path) and os.path.getsize(path) > 1_000_000:
        return path
    os.makedirs(MODEL_DIR, exist_ok=True)
    tmp = path + ".part"
    print(f"Downloading AI model {name} (one time only)...")
    req = urllib.request.Request(MODEL_BASE + name, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out)
    if os.path.getsize(tmp) < 1_000_000:
        os.remove(tmp)
        raise RuntimeError("download incomplete")
    os.replace(tmp, path)
    return path


class OneModel:
    def __init__(self, name, weight, session, size):
        self.name, self.weight, self.session, self.size = name, weight, session, size
        self.input_name = session.get_inputs()[0].name


class EmotionEnsemble:
    """Several trained CNNs (EfficientNet B0 x3 + B2) that each classify a face crop
    into 8 emotions. Their probabilities are averaged (weighted). Every model also
    looks at a mirrored copy of the face (test-time augmentation)."""

    def __init__(self, load=True):
        self.models, self.errors, self.provider = [], [], "CPU"
        if load:
            self._load()

    @staticmethod
    def _make_session(ort, path):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = max(2, (os.cpu_count() or 4) // 2)   # leave CPU for the video
        avail = ort.get_available_providers()
        gpu = [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider") if p in avail]
        for providers in ([*gpu, "CPUExecutionProvider"], ["CPUExecutionProvider"]):
            try:
                sess = ort.InferenceSession(path, sess_options=so, providers=providers)
                shape = sess.get_inputs()[0].shape
                size = shape[2] if isinstance(shape[2], int) else 224
                dummy = np.zeros((1, 3, size, size), np.float32)
                sess.run(None, {sess.get_inputs()[0].name: dummy})        # warm-up / sanity check
                return sess, size, providers[0].replace("ExecutionProvider", "")
            except Exception:
                if providers == ["CPUExecutionProvider"]:
                    raise
        raise RuntimeError("no provider worked")

    def _load(self):
        try:
            import onnxruntime as ort
        except Exception as e:
            self.errors.append(f"onnxruntime missing ({e})")
            print("[Mind Reader] onnxruntime is not installed.  Fix: pip install onnxruntime\n"
                  "              Using simple landmark rules instead.")
            return
        for name, weight in MODEL_SPECS:
            try:
                try:
                    path = download_model(name)
                    sess, size, prov = self._make_session(ort, path)
                except Exception:
                    # a broken / partial file: delete it and try once more
                    bad = os.path.join(MODEL_DIR, name)
                    if os.path.isfile(bad):
                        os.remove(bad)
                    path = download_model(name)
                    sess, size, prov = self._make_session(ort, path)
                out_dim = sess.get_outputs()[0].shape[-1]
                if isinstance(out_dim, int) and out_dim < len(CNN_CLASSES):
                    raise RuntimeError(f"unexpected output size {out_dim}")
                self.models.append(OneModel(name, weight, sess, size))
                self.provider = prov
                print(f"  loaded {name} ({size}x{size}, {prov})")
            except Exception as e:
                self.errors.append(f"{name}: {e}")
                print(f"  [skipped] {name}: {e}")
        if self.models:
            print(f"Emotion AI ready: {len(self.models)} model(s) in the ensemble.")
        else:
            print("[Mind Reader] No AI model could be loaded (check your internet connection).\n"
                  "              Using simple landmark rules instead.")

    @property
    def ready(self):
        return len(self.models) > 0

    def predict(self, face_bgr):
        """Face crop (BGR) -> dict {HAPPY, SURPRISED, SAD, ANGRY, FEAR, DISGUST, NEUTRAL: prob}."""
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        total = np.zeros(len(CNN_CLASSES), np.float64)
        wsum = 0.0
        for m in self.models:
            interp = cv2.INTER_AREA if rgb.shape[0] > m.size else cv2.INTER_LINEAR
            img = cv2.resize(rgb, (m.size, m.size), interpolation=interp).astype(np.float32) / 255.0
            img = (img - IMG_MEAN) / IMG_STD
            batch = np.stack([img, img[:, ::-1]])                       # original + mirrored
            batch = np.ascontiguousarray(batch.transpose(0, 3, 1, 2), dtype=np.float32)
            z = m.session.run(None, {m.input_name: batch})[0][:, :len(CNN_CLASSES)]
            z = z.astype(np.float64) / SOFTMAX_TEMP
            e = np.exp(z - z.max(axis=1, keepdims=True))
            prob = (e / e.sum(axis=1, keepdims=True)).mean(axis=0)
            total += m.weight * prob
            wsum += m.weight
        total /= max(wsum, 1e-9)
        out = {k: 0.0 for k in FACE_KEYS}
        for name, pr in zip(CNN_CLASSES, total):
            out[CNN_TO_MIND[name]] += float(pr)
        return out


class EmotionWorker(threading.Thread):
    """Runs the heavy AI in the background so the camera video never freezes.
    The main loop hands over the newest face crop; results are picked up with poll()."""

    def __init__(self, ensemble):
        super().__init__(daemon=True)
        self.ens = ensemble
        self.cv = threading.Condition()
        self.pending = None
        self.busy = False
        self.stop_flag = False
        self.result, self.seq, self.seen = None, 0, 0
        self.ms = 0.0

    @property
    def idle(self):
        with self.cv:
            return self.pending is None and not self.busy

    def submit(self, crop):
        with self.cv:
            self.pending = crop
            self.cv.notify()

    def poll(self):
        """Newest result if there is one that was not read yet, otherwise None."""
        with self.cv:
            if self.seq == self.seen:
                return None
            self.seen = self.seq
            return dict(self.result)

    def run(self):
        while True:
            with self.cv:
                while self.pending is None and not self.stop_flag:
                    self.cv.wait(0.3)
                if self.stop_flag:
                    return
                crop, self.pending = self.pending, None
                self.busy = True
            probs, ms = None, 0.0
            try:
                t0 = time.time()
                probs = self.ens.predict(crop)
                ms = (time.time() - t0) * 1000.0
            except Exception as e:
                print(f"[Mind Reader] AI prediction failed: {e}")
            with self.cv:
                if probs is not None:
                    self.result, self.ms = probs, ms
                    self.seq += 1
                self.busy = False

    def close(self):
        with self.cv:
            self.stop_flag = True
            self.cv.notify_all()
        if self.is_alive() and threading.current_thread() is not self:
            self.join(timeout=2.0)


BIG_KEYS = {"happy": "HAPPY", "happiness": "HAPPY", "surprised": "SURPRISED", "surprise": "SURPRISED",
            "sad": "SAD", "sadness": "SAD", "angry": "ANGRY", "anger": "ANGRY", "fear": "FEAR",
            "fearful": "FEAR", "disgust": "DISGUST", "disgusted": "DISGUST", "neutral": "NEUTRAL"}


def parse_big_reply(text):
    """Turn the big model's text answer into {HAPPY..NEUTRAL: prob} (or None if unusable)."""
    try:
        a, b = text.index("{"), text.rindex("}")
        data = json.loads(text[a:b + 1])
    except Exception:
        return None
    out = {k: 0.0 for k in FACE_KEYS}
    for key, val in data.items():
        k = BIG_KEYS.get(str(key).strip().lower())
        try:
            v = float(val)
        except Exception:
            continue
        if k and v > 0:
            out[k] += v
    total = sum(out.values())
    if total <= 0:
        return None
    res = {k: v / total for k, v in out.items()}
    th = data.get("thought") if isinstance(data, dict) else None
    if isinstance(th, str):
        th = " ".join(th.split())[:60]
        if th:
            res["_thought"] = th
    return res


class OllamaEmotion:
    """Multi-GB vision-language model (runs locally in Ollama) that looks at your face and
    scores the expression. Slow (seconds), so it runs in the background and is blended with
    the fast ensemble."""

    PROMPT = ("You are an expert in facial expression analysis. Look only at the facial "
              "expression of the person in this photo (do not identify them). "
              "Reply with ONLY a JSON object giving how strongly the face shows each emotion, "
              "as numbers 0-100 that add up to 100, and add a short, fun, harmless guess of what "
              "this person might be thinking (first person, at most 8 words, matching the expression): "
              '{"happy": 0, "surprised": 0, "sad": 0, "angry": 0, "fear": 0, "disgust": 0, "neutral": 0, '
              '"thought": "..."}')

    def __init__(self):
        self.ok, self.last_err = False, ""
        self._check()

    def _check(self):
        try:
            with urllib.request.urlopen(BIG_URL + "/api/tags", timeout=3) as r:
                names = [m.get("name", "") for m in json.load(r).get("models", [])]
        except Exception:
            print(f"[Big model] Ollama is not running at {BIG_URL}. Skipping the big model.\n"
                  "            (Install from ollama.com, open the Ollama app, then restart this program.)")
            return
        if BIG_MODEL in names or BIG_MODEL + ":latest" in names:
            self.ok = True
            print(f"Big AI model ready: {BIG_MODEL}")
        else:
            print(f"[Big model] '{BIG_MODEL}' is not downloaded yet. Run this in Terminal:\n"
                  f"            ollama pull {BIG_MODEL}")

    def predict(self, crop):
        try:
            okj, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            payload = {"model": BIG_MODEL, "stream": False, "format": "json", "keep_alive": "30m",
                       "options": {"temperature": 0.2, "num_predict": 160},
                       "messages": [{"role": "user", "content": self.PROMPT,
                                     "images": [base64.b64encode(buf.tobytes()).decode()]}]}
            req = urllib.request.Request(BIG_URL + "/api/chat", data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=240) as r:
                text = json.load(r).get("message", {}).get("content", "")
            probs = parse_big_reply(text)
            if probs is None:
                self._warn(f"unreadable answer: {text[:80]!r}")
            return probs
        except Exception as e:
            self._warn(str(e))
            return None

    def _warn(self, msg):
        if msg != self.last_err:                 # print each problem only once
            print(f"[Big model] {msg}")
            self.last_err = msg


def hands_cover_face(f, hands):
    """True if a hand is in front of the face (then the AI reading would be wrong)."""
    if not hands:
        return False
    xs = np.array([p.x for p in f.lm]) * f.w
    ys = np.array([p.y for p in f.lm]) * f.h
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    for L in hands:
        inside = np.sum((L[:, 0] > x0) & (L[:, 0] < x1) & (L[:, 1] > y0) & (L[:, 1] < y1))
        if inside >= HAND_COVER_POINTS:
            return True
    return False


def face_crop(frame, f, margin=CROP_MARGIN, size=CROP_SIZE):
    """Upright, square face crop (CROP_SIZE x CROP_SIZE) taken from the clean frame.
    The head tilt is removed by rotating the crop, which helps the AI a lot."""
    h, w = frame.shape[:2]
    pts = np.array([[p.x * w, p.y * h] for p in f.lm], np.float32)
    c = pts.mean(axis=0)
    R = cv2.getRotationMatrix2D((float(c[0]), float(c[1])), f.deg, 1.0)   # levels the head
    rp = pts @ R[:, :2].T + R[:, 2]
    mn, mx = rp.min(axis=0), rp.max(axis=0)
    side = float(max(mx[0] - mn[0], mx[1] - mn[1])) * (1.0 + 2.0 * margin)
    if side < 40:
        return None
    bc = (mn + mx) / 2.0
    s = size / side
    S = np.array([[s, 0, -bc[0] * s + size / 2.0],
                  [0, s, -bc[1] * s + size / 2.0],
                  [0, 0, 1.0]])
    M = (S @ np.vstack([R, [0, 0, 1.0]]))[:2]
    return cv2.warpAffine(frame, M, (size, size), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


class MindReader:
    """Facial-expression reader.
    - Main emotions come from an ensemble of trained neural networks run on the face crop
      (in a background thread).
    - Wink / sleepy / thinking / laughing come from FaceMesh landmark measurements.
    - A short calibration learns your resting face and removes its bias.
    - If the AI models are unavailable, landmark rules score the emotions instead.
    Landmark distances are divided by face width, so distance from camera does not matter."""

    DEFAULT = {"gap": 0.015, "width": 0.34, "lift": 0.0, "ear": 0.27, "ear_l": 0.27,
               "ear_r": 0.27, "ear_diff": 0.0, "brow": 0.075, "yaw": 0.0,
               "pitch": 1.0, "roll": 0.0}

    def __init__(self, load_model=True):
        self.ens = EmotionEnsemble(load=load_model)
        self.worker = EmotionWorker(self.ens) if self.ens.ready else None
        if self.worker is not None:
            self.worker.start()
        self.bigw, self.big_probs, self.big_t = None, None, 0.0
        self.big_thought, self.big_thought_t, self.big_thought_emo = "", -1e9, None
        if load_model and BIG_WEIGHT > 0:
            client = OllamaEmotion()
            if client.ok:
                self.bigw = EmotionWorker(client)
                self.bigw.start()
        self.cnn = None                      # smoothed AI probabilities
        self.ai_bias = None                  # AI probabilities of your resting face
        self.base = dict(self.DEFAULT)
        self.calibrated = False
        self.calibrating = False
        self.cal_t0 = 0.0
        self.samples, self.cal_ai = [], []
        self.votes = deque(maxlen=8)
        self.scores = {k: 0.0 for k in MIND_EMOTIONS}
        self.emotion = "NEUTRAL"
        self.thought = "Reading your mind..."
        self.thought_t0 = 0.0
        self.thought_emotion = None

    # ---- info
    @property
    def ready(self):
        return self.worker is not None

    @property
    def engine(self):
        if self.ready:
            extra = f" + {BIG_MODEL}" if self.bigw is not None else ""
            return f"AI ensemble x{len(self.ens.models)} + mirror{extra}"
        return "landmark rules (no AI model)"

    @property
    def big_can_accept(self):
        return self.bigw is not None and self.bigw.idle

    def feed_big(self, crop):
        if self.bigw is not None and crop is not None:
            self.bigw.submit(crop)

    def big_fresh(self, now):
        if self.big_probs is not None and now - self.big_t < BIG_MAX_AGE:
            return self.big_probs
        return None

    @property
    def ai_ms(self):
        return self.worker.ms if self.worker is not None else 0.0

    # ---- AI plumbing
    @property
    def can_accept(self):
        return self.worker is not None and self.worker.idle

    def feed(self, crop):
        if self.worker is not None and crop is not None:
            self.worker.submit(crop)

    def close(self):
        if self.worker is not None:
            self.worker.close()
        if self.bigw is not None:
            self.bigw.close()

    def reset_live(self):
        """Called when the mind reader is switched off, so old data is not reused."""
        self.cnn = None
        self.big_probs = None
        self.big_thought = ""
        self.votes.clear()
        if self.calibrating:
            self.calibrating = False         # unfinished calibration is redone next time

    # ---- calibration
    def start_calibration(self, now):
        self.samples, self.cal_ai = [], []
        self.calibrating = True
        self.cal_t0 = now

    def progress(self):
        p = len(self.samples) / CAL_FRAMES
        if self.ready:
            p = min(p, len(self.cal_ai) / CAL_AI)
        return clip01(p)

    def _finish_calibration(self):
        if len(self.samples) >= 5:
            keys = self.samples[0].keys()
            self.base = {k: float(np.mean([x[k] for x in self.samples])) for k in keys}
        if self.cal_ai:
            self.ai_bias = {k: min(float(np.mean([a[k] for a in self.cal_ai])), BIAS_CAP)
                            for k in FACE_KEYS}
        self.calibrated = True
        self.calibrating = False
        self.cnn = None
        self.votes.clear()

    def _debias(self, p):
        """Move the part of each emotion that your resting face already shows to NEUTRAL."""
        if self.ai_bias is None:
            return p
        out, removed = {}, 0.0
        for k in FACE_KEYS:
            if k == "NEUTRAL":
                continue
            v = max(p[k] - BIAS_STRENGTH * self.ai_bias[k], 0.0)
            removed += p[k] - v
            out[k] = v
        out["NEUTRAL"] = p["NEUTRAL"] + removed
        return out

    # ---- measurements
    @staticmethod
    def measure(f):
        fw = f.fw
        gap = np.linalg.norm(f.pt(13) - f.pt(14)) / fw
        width = np.linalg.norm(f.pt(61) - f.pt(291)) / fw
        center = f.pt(13)                       # upper lip: does not move when the jaw opens
        corners = (f.pt(61) + f.pt(291)) / 2
        lift = float(np.dot(corners - center, f.u)) / fw          # + when smiling
        ear_r = np.linalg.norm(f.pt(159) - f.pt(145)) / (np.linalg.norm(f.pt(33) - f.pt(133)) + 1e-6)
        ear_l = np.linalg.norm(f.pt(386) - f.pt(374)) / (np.linalg.norm(f.pt(362) - f.pt(263)) + 1e-6)
        brow = (np.linalg.norm(f.pt(105) - f.pt(159)) + np.linalg.norm(f.pt(334) - f.pt(386))) / 2 / fw
        mid = (f.pt(234) + f.pt(454)) / 2
        yaw = float(np.dot(f.pt(1) - mid, f.r)) / fw
        pitch = np.linalg.norm(f.pt(1) - f.pt(10)) / (np.linalg.norm(f.pt(1) - f.pt(152)) + 1e-6)
        return {"gap": float(gap), "width": float(width), "lift": lift,
                "ear": float((ear_l + ear_r) / 2), "ear_l": float(ear_l),
                "ear_r": float(ear_r), "ear_diff": float(ear_r - ear_l),
                "brow": float(brow), "yaw": yaw, "pitch": float(pitch), "roll": f.deg}

    def update(self, f, now, occluded=False):
        m = self.measure(f)
        ai = self.worker.poll() if self.worker is not None else None
        nb = self.bigw.poll() if self.bigw is not None else None
        if nb is not None and not occluded:
            self.big_probs = nb if self.big_probs is None else {
                k: 0.5 * self.big_probs[k] + 0.5 * nb[k] for k in FACE_KEYS}
            self.big_t = now
            th = nb.get("_thought")
            if th:
                self.big_thought, self.big_thought_t = th, now
                self.big_thought_emo = None          # filled in below, once the mood is known
        if self.calibrating:
            if not occluded:
                self.samples.append(m)
                if ai is not None:
                    self.cal_ai.append(ai)
            if self.progress() >= 1.0 or now - self.cal_t0 > CAL_TIMEOUT:
                self._finish_calibration()
            return
        b = self.base
        d = {k: m[k] - b[k] for k in m}
        geo = {
            "HAPPY": clip01(0.6 * d["lift"] / 0.025 + 0.4 * d["width"] / 0.05),
            "SURPRISED": clip01(0.45 * d["brow"] / 0.02 + 0.45 * d["gap"] / 0.06
                                + 0.10 * d["ear"] / 0.08),
            "SAD": clip01(0.8 * clip01(-d["lift"] / 0.02) + 0.2 * clip01(-d["ear"] / 0.08)),
            "ANGRY": clip01(0.75 * clip01(-d["brow"] / 0.015) + 0.25 * clip01(-d["width"] / 0.04)),
            "SLEEPY": clip01(-d["ear"] / 0.12),
            "WINK": (clip01((abs(d["ear_diff"]) - 0.05) / 0.10)
                     if max(m["ear_l"], m["ear_r"]) > 0.6 * b["ear"] else 0.0),
            "THINKING": max(clip01((abs(d["yaw"]) - 0.04) / 0.08),
                            clip01((abs(d["roll"]) - 8) / 14),
                            clip01((abs(d["pitch"]) - 0.12) / 0.2)),
        }
        # ---- main emotions: AI ensemble if available, otherwise landmark rules
        if ai is not None and not occluded:
            ai = self._debias(ai)
            if self.cnn is None:
                self.cnn = ai
            else:
                self.cnn = {k: (1 - AI_ALPHA) * self.cnn[k] + AI_ALPHA * ai[k] for k in FACE_KEYS}
        use_ai = self.ready and self.cnn is not None
        big = self.big_fresh(now)
        if use_ai:
            raw = {k: self.cnn[k] for k in FACE_KEYS}
            if big is not None:                      # blend the fast ensemble with the big model
                raw = {k: (1 - BIG_WEIGHT) * raw[k] + BIG_WEIGHT * big[k] for k in FACE_KEYS}
        elif big is not None:
            raw = dict(big)
            use_ai = True
        else:
            raw = {k: geo.get(k, 0.0) for k in FACE_KEYS}
            raw["NEUTRAL"] = clip01(1.0 - max(raw.values()))
        for k in EXTRA_KEYS:
            raw[k] = geo[k]
        for k in MIND_EMOTIONS:
            self.scores[k] = 0.5 * self.scores[k] + 0.5 * raw[k]
        # ---- decide the current mood
        sc = self.scores
        best = max(FACE_KEYS, key=sc.get)
        cand = best if sc[best] >= (AI_MIN_CONF if use_ai else 0.5) else "NEUTRAL"
        if sc["WINK"] > 0.6:
            cand = "WINK"
        elif sc["SLEEPY"] > 0.75:
            cand = "SLEEPY"
        elif sc["THINKING"] > 0.7 and cand == "NEUTRAL":
            cand = "THINKING"
        if cand == "HAPPY" and d["gap"] > 0.03:
            cand = "LAUGHING"
        self.votes.append(cand)
        name, count = Counter(self.votes).most_common(1)[0]
        if count >= 4:
            self.emotion = name
        # ---- thought bubble: written by the big model when it has one for the current mood
        if self.big_thought and self.big_thought_emo is None:
            self.big_thought_emo = self.emotion
            self.thought, self.thought_t0, self.thought_emotion = self.big_thought, now, self.emotion
        big_thought_ok = (self.big_thought != "" and now - self.big_thought_t < BIG_THOUGHT_TIME
                          and self.big_thought_emo == self.emotion)
        if big_thought_ok:
            pass
        # ---- otherwise pick a built-in thought when the mood changes or every few seconds
        elif self.emotion != self.thought_emotion or now - self.thought_t0 > 4.0:
            pool = THOUGHTS[self.emotion]
            choices = [t for t in pool if t != self.thought] or pool
            self.thought = random.choice(choices)
            self.thought_t0 = now
            self.thought_emotion = self.emotion


def draw_emoji(frame, c, r, emo):
    cx, cy = int(c[0]), int(c[1])
    black = (30, 30, 30)
    cv2.circle(frame, (cx, cy), r, (0, 215, 255), -1)
    cv2.circle(frame, (cx, cy), r, (0, 140, 200), 2)
    ex, ey, er = int(r * 0.38), int(cy - r * 0.2), max(int(r * 0.12), 2)
    my = int(cy + r * 0.38)
    lx, rx = cx - ex, cx + ex

    def eye(x, big=1.0):
        cv2.circle(frame, (x, ey), int(er * big), black, -1)

    def line_eye(x):
        cv2.line(frame, (x - er * 2, ey), (x + er * 2, ey), black, 2)

    def arc_eye(x):
        cv2.ellipse(frame, (x, ey + er), (er * 2, er * 2), 0, 180, 360, black, 2)

    if emo in ("HAPPY", "LAUGHING"):
        arc_eye(lx); arc_eye(rx)
    elif emo in ("SLEEPY", "DISGUST"):
        line_eye(lx); line_eye(rx)
    elif emo == "FEAR":
        eye(lx, 1.5); eye(rx, 1.5)
    elif emo == "WINK":
        eye(lx); line_eye(rx)
    elif emo == "SURPRISED":
        eye(lx, 1.7); eye(rx, 1.7)
    else:
        eye(lx); eye(rx)

    if emo == "HAPPY":
        cv2.ellipse(frame, (cx, my - int(r * 0.1)), (int(r * 0.4), int(r * 0.3)), 0, 10, 170, black, 3)
    elif emo == "WINK":
        cv2.ellipse(frame, (cx, my - int(r * 0.1)), (int(r * 0.35), int(r * 0.25)), 0, 10, 170, black, 3)
    elif emo == "LAUGHING":
        cv2.ellipse(frame, (cx, my - int(r * 0.15)), (int(r * 0.42), int(r * 0.35)), 0, 0, 180, black, -1)
    elif emo == "SURPRISED":
        cv2.ellipse(frame, (cx, my + 2), (int(r * 0.14), int(r * 0.2)), 0, 0, 360, black, -1)
    elif emo == "SAD":
        cv2.ellipse(frame, (cx, my + int(r * 0.25)), (int(r * 0.3), int(r * 0.2)), 0, 190, 350, black, 3)
        cv2.line(frame, (lx - er * 2, ey - er * 3), (lx + er * 2, ey - er * 2 - 3), black, 2)
        cv2.line(frame, (rx + er * 2, ey - er * 3), (rx - er * 2, ey - er * 2 - 3), black, 2)
    elif emo == "ANGRY":
        cv2.line(frame, (cx - int(r * 0.25), my + 4), (cx + int(r * 0.25), my + 4), black, 3)
        cv2.line(frame, (lx - er * 2, ey - er * 4), (lx + er * 2, ey - er), black, 3)
        cv2.line(frame, (rx + er * 2, ey - er * 4), (rx - er * 2, ey - er), black, 3)
    elif emo == "SLEEPY":
        cv2.ellipse(frame, (cx, my + 2), (int(r * 0.1), int(r * 0.12)), 0, 0, 360, black, 2)
        cv2.putText(frame, "z", (cx + r - 4, cy - r + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    elif emo == "THINKING":
        cv2.line(frame, (cx - int(r * 0.1), my), (cx + int(r * 0.3), my - 4), black, 3)
        cv2.line(frame, (rx - er * 2, ey - er * 4), (rx + er * 2, ey - er * 5), black, 2)
    elif emo == "FEAR":
        cv2.ellipse(frame, (cx, my + 2), (int(r * 0.2), int(r * 0.14)), 0, 0, 360, black, 2)
        cv2.line(frame, (lx - er * 2, ey - er * 3), (lx + er * 2, ey - er * 4), black, 2)
        cv2.line(frame, (rx + er * 2, ey - er * 3), (rx - er * 2, ey - er * 4), black, 2)
    elif emo == "DISGUST":
        zig = [(cx - int(r * 0.3), my + 3), (cx - int(r * 0.1), my - 3),
               (cx + int(r * 0.1), my + 3), (cx + int(r * 0.3), my - 3)]
        cv2.polylines(frame, [np.array(zig, np.int32)], False, black, 2, cv2.LINE_AA)
    else:
        cv2.line(frame, (cx - int(r * 0.25), my), (cx + int(r * 0.25), my), black, 3)


def draw_thought(frame, f, text, now_chars):
    """Thought bubble near the head with a typewriter effect."""
    h, w = frame.shape[:2]
    lines = textwrap.wrap(text, 16) or [""]
    scale, th = 0.8, 2
    sizes = [cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, scale, th)[0] for ln in lines]
    tw = max(sz[0] for sz in sizes)
    lh = max(sz[1] for sz in sizes) + 12
    ax, ay = tw // 2 + 34, (lh * len(lines)) // 2 + 28
    head = f.pt(10)
    c = f.at(head, 0.62, 0.55)
    cx = int(np.clip(c[0], ax + 10, max(w - ax - 10, ax + 10)))
    cy = int(np.clip(c[1], ay + 10, max(h - ay - 10, ay + 10)))
    ctr = np.array([cx, cy], dtype=float)

    def draw(o):
        cv2.ellipse(o, (cx, cy), (ax, ay), 0, 0, 360, (255, 255, 255), -1)
        for t, rad in ((0.2, 6), (0.4, 10), (0.62, 15)):
            p = head + (ctr - head) * t
            cv2.circle(o, ip(p), rad, (255, 255, 255), -1)
    alpha_draw(frame, draw, 0.92)
    cv2.ellipse(frame, (cx, cy), (ax, ay), 0, 0, 360, (90, 90, 90), 2, cv2.LINE_AA)
    remaining = now_chars
    y = cy - (lh * len(lines)) // 2
    for ln in lines:
        part = ln[:max(0, min(len(ln), remaining))]
        remaining -= len(ln) + 1
        y += lh
        cv2.putText(frame, part, (cx - tw // 2, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (40, 40, 40), th, cv2.LINE_AA)


def draw_scan(frame, face_lms, f, now, mp_draw, mp_face):
    """Sci-fi face scan: contour mesh + moving scan line."""
    mp_draw.draw_landmarks(frame, face_lms, mp_face.FACEMESH_CONTOURS,
                           landmark_drawing_spec=None,
                           connection_drawing_spec=mp_draw.DrawingSpec(
                               color=(255, 255, 0), thickness=1))
    top, bottom = f.pt(10)[1], f.pt(152)[1]
    y = int(top + ((now * 0.7) % 1.0) * (bottom - top))
    x0, x1 = int(f.pt(234)[0]), int(f.pt(454)[0])
    alpha_draw(frame, lambda o: cv2.line(o, (x0, y), (x1, y), (255, 255, 0), 6), 0.5)


def draw_mind_panel(frame, reader):
    h, w = frame.shape[:2]
    x0, rowh = 20, 22
    box_h = rowh * (len(MIND_EMOTIONS) + 5) + 16
    y0 = h - box_h - 60
    alpha_draw(frame, lambda o: cv2.rectangle(o, (x0 - 10, y0), (x0 + 395, y0 + box_h),
                                              (0, 0, 0), -1), 0.5)
    cv2.putText(frame, "MIND READER", (x0, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 0), 1)
    draw_emoji(frame, (x0 + 348, y0 + 62), 27, reader.emotion)
    for k, name in enumerate(MIND_EMOTIONS):
        y = y0 + 22 + rowh * (k + 1)
        sc = reader.scores[name]
        active = name == reader.emotion or (reader.emotion == "LAUGHING" and name == "HAPPY")
        color = (0, 255, 0) if active else (255, 255, 255)
        cv2.putText(frame, name, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.rectangle(frame, (x0 + 105, y - 12), (x0 + 265, y + 2), (255, 255, 255), 1)
        cv2.rectangle(frame, (x0 + 105, y - 12), (x0 + 105 + int(160 * sc), y + 2), color, -1)
        cv2.putText(frame, f"{int(sc * 100)}%", (x0 + 272, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, color, 1)
    y = y0 + 22 + rowh * (len(MIND_EMOTIONS) + 1)
    cv2.putText(frame, f"Mood: {reader.emotion}", (x0, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"Engine: {reader.engine}", (x0, y + rowh - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200) if reader.ready else (120, 120, 255), 1)
    if reader.bigw is not None:
        age = time.time() - reader.big_t if reader.big_probs is not None else None
        big_txt = (f"Big AI {BIG_MODEL}: " + ("thinking..." if age is None else f"{age:.0f}s ago"))
        cv2.putText(frame, big_txt, (x0, y + 3 * rowh - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 220, 150), 1)
    cal = "calibrated" if reader.calibrated else "not calibrated (press c)"
    ms = f"  |  AI {reader.ai_ms:.0f} ms" if reader.ready else ""
    cv2.putText(frame, f"Neutral face: {cal}{ms}", (x0, y + 2 * rowh - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(frame, "Reads facial expression only - it cannot read real thoughts",
                (20, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)


# ------------------------------------------------------------------ HUD / screenshot
def save_screenshot(clean_frame):
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    name = datetime.now().strftime("mindreader_%Y%m%d_%H%M%S.png")
    path = os.path.join(SCREENSHOT_DIR, name)
    cv2.imwrite(path, clean_frame)
    print(f"Saved {path}")
    return path


def draw_legend(frame, active):
    h, w = frame.shape[:2]
    row = 24
    x0, y0 = w - 350, 10
    rows = [(g, d, e) for g, d, e in GESTURE_INFO] + [
        (None, "Point finger at head", "Mind reader"),
        (None, "Key m", "Mind reader on/off")]
    box_h = row * (len(rows) + 1) + 10
    alpha_draw(frame, lambda o: cv2.rectangle(o, (x0, y0), (w - 10, y0 + box_h),
                                              (0, 0, 0), -1), 0.5)
    cv2.putText(frame, "GESTURE -> EFFECT", (x0 + 10, y0 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    for k, (g, desc, eff) in enumerate(rows):
        color = (0, 255, 0) if g == active else (255, 255, 255)
        cv2.putText(frame, f"{desc} -> {eff}", (x0 + 10, y0 + 22 + row * (k + 1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def draw_debug(frame, states, raw):
    if states is None:
        return
    x, y = 20, 170
    cv2.putText(frame, f"raw: {raw}", (x, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1)
    for k, (key, label) in enumerate([("thumb", "T"), ("index", "I"), ("middle", "M"),
                                      ("ring", "R"), ("pinky", "P")]):
        color = (0, 255, 0) if states[key] else (0, 0, 255)
        cv2.putText(frame, label, (x + k * 32, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    color, 2)


# ------------------------------------------------------------------ main
def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try a different CAM_INDEX.")

    mp_hands = mp.solutions.hands
    mp_face = mp.solutions.face_mesh
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                           min_detection_confidence=0.6, min_tracking_confidence=0.5)
    face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1,
                                 min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)

    stabilizer = GestureStabilizer()
    hearts, confetti = HeartParticles(), Confetti()
    print("Loading Mind Reader (first run downloads about 80 MB of AI models)...")
    mind = MindReader()
    mind_mode, mind_until, cal_request, was_active = False, 0.0, False, False
    show_landmarks, show_debug, show_legend = True, False, True
    test_idx = None
    point_frames = 0
    last_shot = 0.0
    flash = 0.0
    saved_text_until = 0.0
    prev = time.time()
    win = "Mind Reader AR"

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (PROC_W, int(h * PROC_W / w)))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)   # MediaPipe expects RGB
        rgb.flags.writeable = False

        # ---- hand detection (landmarks -> pixel arrays)
        hand_res = hands.process(rgb)
        hand_L = []
        if hand_res.multi_hand_landmarks:
            for hl in hand_res.multi_hand_landmarks:
                hand_L.append(np.array([[p.x * w, p.y * h] for p in hl.landmark]))

        # ---- gesture recognition
        raw, states = NONE, None
        if hand_L:
            results = [classify(L) for L in hand_L]
            states = results[0][1]
            if is_heart(hand_L):
                raw = HEART
            elif len(results) >= 2 and all(g == OPEN_PALM for g, _ in results[:2]):
                raw = CONFETTI
            else:
                raw = next((g for g, _ in results if g != NONE), NONE)
        gesture = stabilizer.update(raw)
        if test_idx is not None:
            gesture = TEST_ORDER[test_idx]

        # ---- face landmarks (only when something needs them)
        mind_active = mind_mode or now < mind_until
        face, face_lms = None, None
        if gesture in FACE_FX or mind_active or gesture == POINTING:
            face_res = face_mesh.process(rgb)
            if face_res.multi_face_landmarks:
                face_lms = face_res.multi_face_landmarks[0]
                face = FaceFrame(face_lms.landmark, w, h)

        # ---- pointing a finger at your head turns the mind reader on
        pointing_head = False
        if gesture == POINTING and face is not None and hand_L:
            tip = hand_L[0][8]
            near = min(np.linalg.norm(tip - face.pt(i)) for i in (10, 21, 251, 67, 297))
            if near < HEAD_TOUCH_RATIO * face.fw:
                pointing_head = True
                mind_until = now + MIND_TIME
                mind_active = True

        if was_active and not mind_active:
            mind.reset_live()
        was_active = mind_active

        # ---- clean face crop for the AI (taken BEFORE any effect is drawn).
        #      Skipped while a hand covers the face, so the AI never reads your hand.
        occluded = bool(mind_active and face is not None and hands_cover_face(face, hand_L))
        if mind_active and face is not None and not occluded and mind.can_accept:
            mind.feed(face_crop(frame, face))
        if mind_active and face is not None and not occluded and mind.big_can_accept:
            mind.feed_big(face_crop(frame, face, size=BIG_CROP))

        # ---- gesture AR effect
        if face is not None and gesture in FACE_FX:
            FACE_FX[gesture](frame, face)

        # ---- particle effects
        if gesture == HEART:
            if len(hand_L) >= 2:
                mx = (hand_L[0][8][0] + hand_L[1][8][0]) / 2
                my = (hand_L[0][8][1] + hand_L[1][8][1]) / 2
            else:
                mx, my = w / 2, h * 0.7
            for _ in range(2):
                hearts.spawn(mx, my)
        if gesture == CONFETTI:
            confetti.spawn(w)
        hearts.update_and_draw(frame)
        confetti.update_and_draw(frame)

        # ---- mind reader
        if mind_active and face is not None:
            if cal_request or (not mind.calibrated and not mind.calibrating):
                mind.start_calibration(now)
                cal_request = False
            mind.update(face, now, occluded)
            draw_scan(frame, face_lms, face, now, mp_draw, mp_face)
            if mind.calibrating:
                text = "Look at the camera with a neutral face..."
                draw_thought(frame, face, text, len(text))
                bar = int(300 * mind.progress())
                cv2.rectangle(frame, (w // 2 - 150, h - 90), (w // 2 + 150, h - 75), (255, 255, 255), 1)
                cv2.rectangle(frame, (w // 2 - 150, h - 90), (w // 2 - 150 + bar, h - 75), (0, 255, 0), -1)
            else:
                chars = int((now - mind.thought_t0) * 28)
                draw_thought(frame, face, mind.thought, chars)

        # ---- screenshot gesture (hold pointing; not when pointing at head)
        point_frames = point_frames + 1 if (gesture == POINTING and not pointing_head) else 0
        if point_frames >= SCREENSHOT_HOLD and now - last_shot > SCREENSHOT_COOLDOWN:
            save_screenshot(frame)
            last_shot, flash, saved_text_until = now, 1.0, now + 1.5
            point_frames = 0

        clean = frame.copy()

        # ---- HUD
        if show_landmarks:
            for hl in hand_res.multi_hand_landmarks or []:
                mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now
        cv2.putText(frame, f"Gesture: {gesture}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Effect: {EFFECT_NAMES[gesture]}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.0f}", (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if test_idx is not None:
            cv2.putText(frame, "TEST MODE (n = next, 0 = auto)", (20, 145),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        if show_debug:
            draw_debug(frame, states, raw)
        if mind_active:
            draw_mind_panel(frame, mind)
            if face is None:
                cv2.putText(frame, "Show your face to the camera", (20, 260),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif occluded:
                cv2.putText(frame, "Move your hand away from your face", (20, 260),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        if show_legend:
            draw_legend(frame, gesture)
        cv2.putText(frame, "q quit | s shot | h landmarks | d debug | g legend | m mind | c calibrate | n test | 0 auto",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if gesture == POINTING and point_frames > 0:
            bar = int(200 * point_frames / SCREENSHOT_HOLD)
            cv2.rectangle(frame, (20, 230), (220, 245), (255, 255, 255), 1)
            cv2.rectangle(frame, (20, 230), (20 + bar, 245), (0, 255, 0), -1)
        if now < saved_text_until:
            cv2.putText(frame, "Screenshot saved!", (w // 2 - 150, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
        if flash > 0:
            cv2.addWeighted(np.full_like(frame, 255), flash, frame, 1 - flash, 0, frame)
            flash -= 0.1

        cv2.imshow(win, frame)
        key = cv2.waitKey(1) & 0xFF
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:   # window closed with the X button
            break
        if key == ord("q"):
            break
        elif key == ord("s"):
            save_screenshot(clean)
            flash, saved_text_until = 1.0, time.time() + 1.5
        elif key == ord("h"):
            show_landmarks = not show_landmarks
        elif key == ord("d"):
            show_debug = not show_debug
        elif key == ord("g"):
            show_legend = not show_legend
        elif key == ord("m"):
            mind_mode = not mind_mode
        elif key == ord("c"):
            cal_request = True
            mind_mode = True
        elif key == ord("n"):
            test_idx = 0 if test_idx is None else (test_idx + 1) % len(TEST_ORDER)
        elif key == ord("0"):
            test_idx = None

    mind.close()
    hands.close()
    face_mesh.close()
    cap.release()
    cv2.destroyAllWindows()
    sys.stdout.flush()
    os._exit(0)          # exit at once, so a background AI thread can never cause a crash message


if __name__ == "__main__":
    main()