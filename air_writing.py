"""
Air Writing AI - real-time air writing with computer vision
Stack: Python, OpenCV, MediaPipe, NumPy

Controls:
    d : toggle drawing on/off
    c : clear canvas
    s : save the current writing as a 28x28 image (for a future classifier)
    q : quit

Install:
    pip install opencv-python mediapipe numpy
"""

import time
from dataclasses import dataclass, field

import cv2
import mediapipe as mp
import numpy as np

# ---------------------------------------------------------------- settings
CAM_INDEX = 0
FRAME_W, FRAME_H = 1280, 720
SMOOTHING_ALPHA = 0.4      # 0..1, lower = smoother but more lag
PEN_UP_PATIENCE = 4        # frames of "not writing" before a stroke ends
MIN_STROKE_POINTS = 2
LINE_THICKNESS = 6
WRITE_COLOR = (0, 255, 0)  # BGR
IDLE_COLOR = (0, 0, 255)
STROKE_COLOR = (255, 0, 255)

# MediaPipe landmark indices: (tip, pip)
FINGERS = {
    "index": (8, 6),
    "middle": (12, 10),
    "ring": (16, 14),
    "pinky": (20, 18),
}
INDEX_TIP = 8


# ---------------------------------------------------------------- helpers
@dataclass
class Stroke:
    points: list = field(default_factory=list)


class Smoother:
    """Exponential moving average to reduce jitter in the fingertip path."""

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.value = None

    def update(self, point):
        p = np.array(point, dtype=np.float32)
        if self.value is None:
            self.value = p
        else:
            self.value = self.alpha * p + (1 - self.alpha) * self.value
        return int(self.value[0]), int(self.value[1])

    def reset(self):
        self.value = None


def fingers_extended(landmarks):
    """Return dict finger -> bool. Extended when tip is above PIP (tip.y < pip.y).
    Thumb is ignored on purpose: it moves sideways, so the y-test fails."""
    return {
        name: landmarks[tip].y < landmarks[pip].y
        for name, (tip, pip) in FINGERS.items()
    }


def is_writing_gesture(state):
    """Index up; middle, ring and pinky folded."""
    return (
        state["index"]
        and not state["middle"]
        and not state["ring"]
        and not state["pinky"]
    )


def to_pixels(landmark, width, height):
    """Normalized (0-1) -> pixel coordinates."""
    x = min(max(int(landmark.x * width), 0), width - 1)
    y = min(max(int(landmark.y * height), 0), height - 1)
    return x, y


def render_strokes(canvas, strokes, color=STROKE_COLOR, thickness=LINE_THICKNESS):
    """Draw lines only inside each stroke, never between strokes."""
    for stroke in strokes:
        pts = stroke.points
        for i in range(1, len(pts)):
            cv2.line(canvas, pts[i - 1], pts[i], color, thickness)


def strokes_to_28x28(strokes, size=28, pad=4):
    """Crop to the writing's bounding box and resize to a square image.
    Output is white-on-black, MNIST/EMNIST style, ready for a classifier."""
    all_pts = [p for s in strokes for p in s.points]
    if len(all_pts) < 2:
        return None
    xs, ys = zip(*all_pts)
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    w, h = max(x1 - x0, 1), max(y1 - y0, 1)
    side = max(w, h) + 40
    img = np.zeros((side, side), dtype=np.uint8)
    ox = (side - w) // 2 - x0
    oy = (side - h) // 2 - y0
    for s in strokes:
        for i in range(1, len(s.points)):
            a = (s.points[i - 1][0] + ox, s.points[i - 1][1] + oy)
            b = (s.points[i][0] + ox, s.points[i][1] + oy)
            cv2.line(img, a, b, 255, max(side // 14, 2))
    inner = size - 2 * pad
    img = cv2.resize(img, (inner, inner), interpolation=cv2.INTER_AREA)
    return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)


# ---------------------------------------------------------------- main
def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try a different CAM_INDEX.")

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,       # track between frames instead of re-detecting
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )

    strokes: list[Stroke] = []
    current: Stroke | None = None
    smoother = Smoother(SMOOTHING_ALPHA)
    drawing_enabled = True
    missed_frames = 0
    prev_time = time.time()
    save_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)                       # mirror view
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)     # MediaPipe wants RGB
        rgb.flags.writeable = False
        result = hands.process(rgb)

        writing = False
        tip_px = None

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            lm = hand.landmark
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            state = fingers_extended(lm)
            writing = drawing_enabled and is_writing_gesture(state)
            tip_px = smoother.update(to_pixels(lm[INDEX_TIP], w, h))

        # ---- stroke management
        if writing and tip_px is not None:
            missed_frames = 0
            if current is None:
                current = Stroke()
            current.points.append(tip_px)
        else:
            missed_frames += 1
            if current is not None and missed_frames >= PEN_UP_PATIENCE:
                if len(current.points) >= MIN_STROKE_POINTS:
                    strokes.append(current)
                current = None
                smoother.reset()

        # ---- rendering
        render_strokes(frame, strokes)
        if current is not None:
            render_strokes(frame, [current])

        if tip_px is not None:
            color = WRITE_COLOR if writing else IDLE_COLOR
            cv2.circle(frame, tip_px, 10, color, cv2.FILLED)

        label = "Writing" if writing else "Pen up"
        if not drawing_enabled:
            label = "Drawing OFF"
        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    WRITE_COLOR if writing else IDLE_COLOR, 2)

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(frame, f"FPS: {fps:.0f}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "d: toggle  c: clear  s: save  q: quit",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1)

        cv2.imshow("Air Writing AI", frame)

        # ---- keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            strokes.clear()
            current = None
            smoother.reset()
        elif key == ord("d"):
            drawing_enabled = not drawing_enabled
        elif key == ord("s"):
            all_strokes = strokes + ([current] if current else [])
            img = strokes_to_28x28(all_strokes)
            if img is not None:
                name = f"airwriting_{save_count:03d}.png"
                cv2.imwrite(name, img)
                print(f"Saved {name}")
                save_count += 1

    hands.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()