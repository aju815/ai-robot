"""
Hand Gesture Air Drawing
========================
Draw on the screen in the air using your index finger. No mouse, no touch.

Install:
    pip install opencv-python mediapipe numpy

Run:
    python hand_gesture_drawing.py

Gestures:
    - Index finger UP only            -> DRAW
    - Index + Middle fingers UP       -> SELECT MODE (hover over the top palette
                                         to pick a colour, eraser or clear)
    - Any other gesture               -> IDLE (no drawing)

Keyboard:
    c -> clear canvas
    s -> save drawing as PNG
    q or ESC -> quit
"""

import time
import cv2
import numpy as np
import mediapipe as mp

# ----------------------------- Settings ------------------------------------
CAM_INDEX = 0
FRAME_W, FRAME_H = 1280, 720
HEADER_H = 110            # height of the colour palette bar
BRUSH_THICKNESS = 8
ERASER_THICKNESS = 50
SMOOTHING = 0.5           # 0 = no smoothing, closer to 1 = smoother but laggier

# Palette: (label, BGR colour). "Eraser" and "Clear" are special actions.
PALETTE = [
    ("Blue",   (255, 80, 0)),
    ("Green",  (0, 200, 0)),
    ("Red",    (0, 0, 255)),
    ("Yellow", (0, 230, 255)),
    ("Pink",   (203, 120, 255)),
    ("Eraser", (0, 0, 0)),
    ("Clear",  (60, 60, 60)),
]

# ------------------------- MediaPipe setup ---------------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.6,
)

# Landmark indices for fingertips and the joints below them (PIP)
TIP_IDS = [4, 8, 12, 16, 20]
PIP_IDS = [3, 6, 10, 14, 18]


def fingers_up(landmarks, handedness_label):
    """Return [thumb, index, middle, ring, pinky] as 1 (up) or 0 (down)."""
    fingers = []

    # Thumb: compare x positions (depends on left/right hand)
    # The frame is mirrored, so labels are already flipped correctly.
    if handedness_label == "Right":
        fingers.append(1 if landmarks[TIP_IDS[0]].x < landmarks[PIP_IDS[0]].x else 0)
    else:
        fingers.append(1 if landmarks[TIP_IDS[0]].x > landmarks[PIP_IDS[0]].x else 0)

    # Other four fingers: tip above the PIP joint means finger is up
    for tip, pip in zip(TIP_IDS[1:], PIP_IDS[1:]):
        fingers.append(1 if landmarks[tip].y < landmarks[pip].y else 0)

    return fingers


def draw_palette(frame, selected_index):
    """Draw the colour buttons across the top of the frame."""
    n = len(PALETTE)
    btn_w = FRAME_W // n
    for i, (label, color) in enumerate(PALETTE):
        x1, x2 = i * btn_w, (i + 1) * btn_w
        cv2.rectangle(frame, (x1, 0), (x2, HEADER_H), color, cv2.FILLED)
        # border (thicker if selected)
        border = 6 if i == selected_index else 1
        cv2.rectangle(frame, (x1, 0), (x2, HEADER_H), (255, 255, 255), border)

        text_color = (0, 0, 0) if label in ("Yellow", "Pink") else (255, 255, 255)
        if label in ("Eraser", "Clear"):
            text_color = (255, 255, 255)
        size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        tx = x1 + (btn_w - size[0]) // 2
        ty = (HEADER_H + size[1]) // 2
        cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, text_color, 2, cv2.LINE_AA)


def get_button_index(x):
    """Which palette button is the x-coordinate over?"""
    btn_w = FRAME_W // len(PALETTE)
    return min(x // btn_w, len(PALETTE) - 1)


def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    if not cap.isOpened():
        print("Could not open webcam. Check CAM_INDEX or camera permissions.")
        return

    canvas = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    selected = 0                       # index in PALETTE
    draw_color = PALETTE[selected][1]
    prev_x, prev_y = 0, 0
    smooth_x, smooth_y = 0, 0
    prev_time = 0
    mode_text = "Idle"

    print("Running... press 'q' to quit, 'c' to clear, 's' to save.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)                       # mirror view
        frame = cv2.resize(frame, (FRAME_W, FRAME_H))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        mode_text = "Idle"

        if result.multi_hand_landmarks:
            hand_lms = result.multi_hand_landmarks[0]
            label = result.multi_handedness[0].classification[0].label
            lm = hand_lms.landmark

            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

            # Index fingertip position in pixels
            raw_x = int(lm[8].x * FRAME_W)
            raw_y = int(lm[8].y * FRAME_H)

            # Smooth the movement to reduce jitter
            if smooth_x == 0 and smooth_y == 0:
                smooth_x, smooth_y = raw_x, raw_y
            else:
                smooth_x = int(smooth_x * SMOOTHING + raw_x * (1 - SMOOTHING))
                smooth_y = int(smooth_y * SMOOTHING + raw_y * (1 - SMOOTHING))
            x, y = smooth_x, smooth_y

            f = fingers_up(lm, label)
            index_up, middle_up, ring_up, pinky_up = f[1], f[2], f[3], f[4]

            # ---------------- SELECT MODE: index + middle up ----------------
            if index_up and middle_up and not ring_up and not pinky_up:
                mode_text = "Select"
                prev_x, prev_y = 0, 0       # lift the pen

                cv2.circle(frame, (x, y), 18, draw_color, cv2.FILLED)
                cv2.circle(frame, (x, y), 18, (255, 255, 255), 2)

                if y < HEADER_H:
                    idx = get_button_index(x)
                    name, color = PALETTE[idx]
                    if name == "Clear":
                        canvas[:] = 0
                    else:
                        selected = idx
                        draw_color = color

            # ---------------- DRAW MODE: only index up ----------------------
            elif index_up and not middle_up and not ring_up and not pinky_up:
                is_eraser = PALETTE[selected][0] == "Eraser"
                mode_text = "Erase" if is_eraser else "Draw"

                thickness = ERASER_THICKNESS if is_eraser else BRUSH_THICKNESS
                cursor_color = (255, 255, 255) if is_eraser else draw_color
                cv2.circle(frame, (x, y), thickness // 2 + 4, cursor_color, 2)

                # Don't draw inside the palette area
                if y > HEADER_H:
                    if prev_x == 0 and prev_y == 0:
                        prev_x, prev_y = x, y
                    cv2.line(canvas, (prev_x, prev_y), (x, y),
                             draw_color, thickness, cv2.LINE_AA)
                    prev_x, prev_y = x, y
                else:
                    prev_x, prev_y = 0, 0

            # ---------------- Any other gesture: idle -----------------------
            else:
                prev_x, prev_y = 0, 0
        else:
            prev_x, prev_y = 0, 0
            smooth_x, smooth_y = 0, 0

        # ------------- Merge canvas with the live camera frame --------------
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, inv = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
        inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
        frame = cv2.bitwise_and(frame, inv)
        frame = cv2.bitwise_or(frame, canvas)

        draw_palette(frame, selected)

        # FPS + mode info
        now = time.time()
        fps = 1 / (now - prev_time) if prev_time else 0
        prev_time = now
        cv2.putText(frame, f"Mode: {mode_text}", (20, FRAME_H - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {int(fps)}", (20, FRAME_H - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)

        cv2.imshow("Hand Gesture Air Drawing", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("c"):
            canvas[:] = 0
        elif key == ord("s"):
            filename = f"drawing_{int(time.time())}.png"
            # Save drawing on a white background
            white = np.full_like(canvas, 255)
            mask = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) > 20
            white[mask] = canvas[mask]
            cv2.imwrite(filename, white)
            print(f"Saved {filename}")

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()