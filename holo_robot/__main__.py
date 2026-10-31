"""Run the HOLO-HUMAN avatar with a live webcam background (AR hologram) and hand tracking."""
from __future__ import annotations

import argparse
import os
import sys
import textwrap
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .controls import (HAND_CONNECTIONS, SamGestureController, hand_from_landmarks,
                       update_two_hand_zoom)
from .gltf_loader import load_glb
from .mathutil import OneEuroFilter
from .ollama import OllamaGuide, describe_part
from .procedural import create_humanoid
from .render_gl import GLRenderer
from .rig import Animator, Rig
from .scene import RenderState, Scene, SceneCamera

APP_VERSION = "9.0"
FONT = cv2.FONT_HERSHEY_SIMPLEX
CYAN = (255, 220, 80)       # BGR
WHITE = (255, 255, 255)


def _open_camera(device_index: int):
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    return cv2.VideoCapture(device_index, backend)


def _list_cameras(max_devices: int = 6) -> list[int]:
    available = []
    for device_index in range(max_devices):
        capture = _open_camera(device_index)
        try:
            if capture.isOpened():
                available.append(device_index)
        finally:
            capture.release()
    return available


def _arguments(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the SAM-X1 gesture-controlled holographic humanoid."
    )
    parser.add_argument("model", type=Path, nargs="?",
                        help="optional humanoid .glb model (uses built-in robot if omitted)")
    parser.add_argument("--animation", action="append", type=Path, default=[],
                        help="optional GLB animation donor (may be repeated)")
    parser.add_argument("--camera", type=int, default=0, help="webcam device index (default: 0)")
    parser.add_argument("--list-cameras", action="store_true",
                        help="list camera device indexes and exit")
    parser.add_argument("--model-complexity", type=int, choices=(0, 1), default=0,
                        help="MediaPipe accuracy/performance preset (default: faster 0)")
    parser.add_argument("--ollama-model", default=os.environ.get(
        "HOLO_OLLAMA_MODEL", os.environ.get("MIND_BIG_MODEL", "qwen2.5vl:7b")),
        help="installed local Ollama model for part explanations")
    parser.add_argument("--ollama-url", default=os.environ.get(
        "HOLO_OLLAMA_URL", os.environ.get("OLLAMA_URL", "http://localhost:11434")),
        help="Ollama server URL")
    parser.add_argument("--no-camera", action="store_true",
                        help="show the avatar without opening a webcam or hand tracking")
    parser.add_argument("--no-video", action="store_true",
                        help="track hands but do not show the webcam image (robot on black)")
    parser.add_argument("--video-dim", type=float, default=0.55,
                        help="brightness of the webcam background, 0..1 (default: 0.55)")
    parser.add_argument("--show-demo-objects", action="store_true",
                        help="also show the floating sample objects around the robot")
    parser.add_argument("--width", type=int, default=1280, help="preview width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="preview height (default: 720)")
    args = parser.parse_args(argv)
    if args.model is not None and not args.model.is_file():
        parser.error(
            f"model file not found: {args.model}. Replace the example path with the "
            "location of your humanoid .glb file, or omit the model argument to use "
            "the built-in robot."
        )
    for animation_path in args.animation:
        if not animation_path.is_file():
            parser.error(f"animation file not found: {animation_path}")
    return args


def _shader_region(region_name: str) -> int:
    if "HEAD UNIT" in region_name:
        return 0
    if "TORSO CORE" in region_name:
        return 1
    if "LEFT ARM" in region_name:
        return 2
    if "RIGHT ARM" in region_name:
        return 3
    if "LEFT LEG" in region_name:
        return 4
    if "RIGHT LEG" in region_name:
        return 5
    return -1


# ------------------------------------------------------------------ drawing helpers
def _compose_background(frame: Optional[np.ndarray], width: int, height: int,
                        dim: float) -> np.ndarray:
    """Dimmed, slightly cool-tinted webcam image so the hologram glows on top of it."""
    if frame is None:
        return np.zeros((height, width, 3), np.uint8)
    if frame.shape[1] != width or frame.shape[0] != height:
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    tint = np.array([1.08, 1.0, 0.88], np.float32)          # BGR: a little blue
    out = frame.astype(np.float32) * dim * tint
    # soft vignette to focus the eye on the centre
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    r = np.sqrt(((xx - width / 2) / (width * 0.75)) ** 2 + ((yy - height / 2) / (height * 0.75)) ** 2)
    out *= np.clip(1.15 - r * 0.6, 0.4, 1.0)[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def _draw_hands(image, landmarks_by_side, width, height):
    for side, pts in landmarks_by_side.items():
        px = [(int(x * width), int(y * height)) for x, y in pts]
        for a, b in HAND_CONNECTIONS:
            cv2.line(image, px[a], px[b], CYAN, 2, cv2.LINE_AA)
        for p in px:
            cv2.circle(image, p, 3, WHITE, -1, cv2.LINE_AA)
        cv2.putText(image, side.upper(), (px[0][0] - 20, px[0][1] + 28),
                    FONT, 0.5, CYAN, 1, cv2.LINE_AA)


def _put_text(image, text, org, scale=0.55, color=WHITE, thickness=1):
    cv2.putText(image, text, org, FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def _draw_hud(image, controller, guide, fps, hover_name, show_hands, show_video, has_camera):
    h, w = image.shape[:2]
    panel = image.copy()
    cv2.rectangle(panel, (14, 14), (440, 118), (20, 12, 0), -1)
    image[:] = cv2.addWeighted(panel, 0.55, image, 0.45, 0)
    cv2.rectangle(image, (14, 14), (440, 118), CYAN, 1, cv2.LINE_AA)
    _put_text(image, f"HOLO ROBOT v{APP_VERSION}   {fps:4.0f} FPS", (26, 38), 0.55, CYAN, 1)
    _put_text(image, f"MODE: {controller.mode_name}", (26, 62), 0.6, WHITE, 1)
    y = 84
    for line in textwrap.wrap(controller.instructions, 52)[:2]:
        _put_text(image, line, (26, y), 0.42, (220, 220, 220))
        y += 18
    _put_text(image, f"SELECTED: {controller.selected_part[:48]}", (26, 108), 0.4, CYAN)

    if controller.pointer is not None:
        px, py = controller.pointer
        color = (80, 255, 120) if controller.hand and controller.hand["pinch"] else CYAN
        cv2.circle(image, (px, py), 14, color, 2, cv2.LINE_AA)
        cv2.circle(image, (px, py), 3, color, -1, cv2.LINE_AA)
        if hover_name:
            _put_text(image, hover_name[:48], (px + 20, py - 10), 0.45, color)

    text = guide.text or guide.status
    lines = []
    for paragraph in text.splitlines():
        lines += textwrap.wrap(paragraph, 70) or [""]
    lines = lines[:9]
    if lines:
        top = h - 20 - 20 * len(lines)
        panel = image.copy()
        cv2.rectangle(panel, (14, top - 24), (560, h - 12), (20, 12, 0), -1)
        image[:] = cv2.addWeighted(panel, 0.55, image, 0.45, 0)
        cv2.rectangle(image, (14, top - 24), (560, h - 12), CYAN, 1, cv2.LINE_AA)
        _put_text(image, f"AI GUIDE: {guide.part}"[:60] if guide.part else "AI GUIDE",
                  (26, top - 6), 0.45, CYAN)
        for i, line in enumerate(lines):
            _put_text(image, line, (26, top + 16 + i * 20), 0.45)

    keys = "Q quit | S screenshot | +/- zoom | 0 reset | F focus | I explain"
    if has_camera:
        keys += f" | V video:{'ON' if show_video else 'OFF'} | H hands:{'ON' if show_hands else 'OFF'}"
    _put_text(image, keys, (w - 20 - 8 * len(keys) // 1, h - 18) if False else (w - 760, 28), 0.4,
              (200, 200, 200))


def main(argv: Optional[list[str]] = None) -> int:
    args = _arguments(argv)
    if args.list_cameras:
        devices = _list_cameras()
        if devices:
            print("Available camera indexes: " + ", ".join(map(str, devices)))
        else:
            print(
                "No camera opened. On macOS, allow Camera access for Terminal or "
                "Visual Studio Code in System Settings > Privacy & Security > Camera, "
                "then try again."
            )
        return 0
    if args.width <= 0 or args.height <= 0:
        raise ValueError("preview width and height must be positive")

    model = load_glb(str(args.model)) if args.model is not None else create_humanoid()
    if not model.primitives:
        raise ValueError(f"{args.model}: the GLB contains no triangle meshes")
    rig = Rig(model)
    if not rig.has_humanoid:
        raise ValueError(
            f"{args.model}: no supported humanoid skeleton found; use a Mixamo-compatible "
            "GLB with hips, head, and left/right arm, forearm, and hand bones"
        )

    animator = Animator(rig)
    animator.add_clips(model)
    for animation_path in args.animation:
        animator.add_clips(load_glb(str(animation_path)), tag=f"{animation_path.stem}_")
    if animator.names():
        idle = animator.find("idle", "stand") or animator.names()[0]
        animator.play(idle)

    camera = SceneCamera(args.width, args.height)
    scene = Scene(camera)
    if not args.show_demo_objects:
        scene.objects = []
    renderer = GLRenderer(args.width, args.height)
    capture = None
    hands = None
    controller = SamGestureController()
    guide = OllamaGuide(args.ollama_url, args.ollama_model)
    show_video = not args.no_video
    show_hands = True
    video_dim = float(np.clip(args.video_dim, 0.0, 1.0))
    try:
        renderer.load_rig(rig)
        if not args.no_camera:
            capture = _open_camera(args.camera)
            if not capture.isOpened():
                capture.release()
                raise RuntimeError(
                    f"cannot open camera {args.camera}. On macOS, allow Camera access for "
                    "Terminal or Visual Studio Code in System Settings > Privacy & Security "
                    "> Camera. To find camera indexes, run `python3 -m holo_robot "
                    "--list-cameras`; then retry with `--camera N`."
                )
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            try:
                import mediapipe as mp
            except ImportError as exc:
                raise RuntimeError(
                    "MediaPipe is required for webcam tracking; install "
                    "holo_robot/requirements.txt or use --no-camera"
                ) from exc
            hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=args.model_complexity,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.6,
            )

        smooth = {"left": {}, "right": {}}
        previous_two_hand_span = None
        target_filters = {
            side: OneEuroFilter(min_cutoff=2.0, beta=0.04)
            for side in ("left", "right")
        }
        start = previous = time.perf_counter()
        fps = 60.0
        window = f"HOLO ROBOT 3D v{APP_VERSION}  |  AR HOLOGRAM"
        print(
            f"{window}\n"
            f"Launcher: {Path(__file__).resolve()}\n"
            + ("Camera tracking is OFF (--no-camera)." if args.no_camera else
               f"Camera {args.camera} is ON: your live image is shown behind the hologram "
               "(press V to toggle)."),
            flush=True,
        )
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

        def on_mouse(event, _x, _y, flags, _param):
            if event == cv2.EVENT_MOUSEWHEEL:
                delta = (flags >> 16) & 0xFFFF
                if delta >= 0x8000:
                    delta -= 0x10000
                camera.zoom(0.9 if delta > 0 else 1.1)

        cv2.setMouseCallback(window, on_mouse)
        while True:
            now = time.perf_counter()
            dt = min(now - previous, 0.1)
            elapsed = now - start
            fps = 0.9 * fps + 0.1 / max(now - previous, 1e-3)
            previous = now

            frame = None
            landmarks_by_side = {}
            detected_hands = {}
            if capture is not None:
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"failed to read from webcam device {args.camera}")
                frame = cv2.flip(frame, 1)
                result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                for points, handedness in zip(
                    result.multi_hand_landmarks or [],
                    result.multi_handedness or [],
                ):
                    side = handedness.classification[0].label.lower()
                    if side not in smooth:
                        continue
                    hand = hand_from_landmarks(points.landmark, smooth[side])
                    detected_hands[side] = hand
                    landmarks_by_side[side] = [(point.x, point.y) for point in points.landmark]
                for side in smooth:
                    if side not in detected_hands:
                        smooth[side].clear()

            if animator.current is None:
                rig.reset_pose()
                rig.update_world()
            else:
                animator.update(dt)

            previous_part = controller.selected_part
            gesture_hand = detected_hands.get("right") or detected_hands.get("left")
            controller.update(gesture_hand, scene, elapsed)
            if controller.selected_part != previous_part:
                guide.request(controller.selected_part, describe_part(controller.selected_part))
            previous_two_hand_span = update_two_hand_zoom(
                detected_hands, camera, previous_two_hand_span
            )
            controller.apply_pose(rig)
            robot_model = scene.robot_matrix(rig.fit_matrix())
            if controller.mode != 0:
                inverse_robot = np.linalg.inv(robot_model)
                for side, hand in detected_hands.items():
                    target_world = np.array([
                        (hand["x"] - 0.5) * 2.0,
                        0.20 + (1.0 - hand["y"]) * 1.55,
                        0.20,
                    ], np.float32)
                    target_world = target_filters[side](target_world, now)
                    target_model = (inverse_robot @ np.append(target_world, 1.0))[:3]
                    rig.aim_arm(side, target_model, 0.9)
                    rig.curl_fingers(side, hand["curls"], 0.9)
            scene.update(dt, elapsed)
            scene.regions = [
                (name, (robot_model @ np.append(center, 1.0))[:3], radius * rig.fit_scale)
                for name, center, radius in rig.region_centers()
            ]
            names_used = set()
            for primitive in model.primitives:
                node_index = primitive.node
                region_id = int(rig.region_of_node[node_index])
                if region_id < 0:
                    continue
                node_name = model.nodes[node_index].name.strip().upper()
                if not node_name or node_name in names_used:
                    node_name = f"{node_name or 'ROBOT PART'} {node_index + 1}"
                names_used.add(node_name)
                category = ("HEAD UNIT", "TORSO CORE", "LEFT ARM", "RIGHT ARM",
                            "LEFT LEG", "RIGHT LEG")[region_id]
                center_model = rig.node_pos(node_index)
                center_world = (robot_model @ np.append(center_model, 1.0))[:3]
                world_scale = np.linalg.norm(rig.world[node_index][:3, :3], axis=0)
                radius = max(float(np.max(world_scale)) * 0.25, rig.height_model * 0.004)
                scene.regions.append((f"{category} / {node_name}", center_world,
                                      radius * rig.fit_scale))
            renderer.update_meshes([rig.skin_primitive(prim) for prim in model.primitives])
            hover_region = -1
            hover_name = ""
            if controller.hover is not None and controller.hover[0] == "region":
                hover_name = scene.regions[controller.hover[1]][0]
                hover_region = _shader_region(hover_name)
            selected_region = -1
            if 0 <= controller.selected_region < len(scene.regions):
                selected_region = _shader_region(
                    scene.regions[controller.selected_region][0]
                )
            state = RenderState(
                time=elapsed,
                camera=camera,
                robot_model=robot_model,
                floor_model=np.eye(4, dtype=np.float32),
                objects=scene.objects,
                hover_region=hover_region,
                selected_region=selected_region,
                hover_amt=0.9 if hover_region >= 0 else 0.0,
                selected_amt=0.65 if selected_region >= 0 else 0.0,
                alpha=0.65 + 0.3 * controller.values["core"],
                standby=1.0 if controller.mode == 0 else 0.0,
            )
            hologram = renderer.render(state)
            if hologram.shape != (args.height, args.width, 3) or hologram.dtype != np.uint8:
                raise RuntimeError(
                    f"renderer returned invalid preview {hologram.shape} ({hologram.dtype})"
                )

            # AR composite: dimmed live camera + additive hologram light on top
            background = _compose_background(
                frame if show_video else None, args.width, args.height, video_dim)
            preview = cv2.add(background, hologram)
            if show_hands and landmarks_by_side:
                _draw_hands(preview, landmarks_by_side, args.width, args.height)
            _draw_hud(preview, controller, guide, fps, hover_name,
                      show_hands, show_video, capture is not None)

            cv2.imshow(window, np.ascontiguousarray(preview))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                name = f"holo_robot_{int(time.time())}.png"
                if not cv2.imwrite(name, preview):
                    raise OSError(f"could not save screenshot to {name}")
                print("saved", name)
            elif key in (ord("+"), ord("=")):
                camera.zoom(0.88)
            elif key in (ord("-"), ord("_")):
                camera.zoom(1.14)
            elif key == ord("0"):
                camera.reset()
            elif key == ord("v"):
                show_video = not show_video
            elif key == ord("h"):
                show_hands = not show_hands
            elif key == ord("f"):
                if controller.hover is not None and controller.hover[0] == "region":
                    camera.focus(scene.regions[controller.hover[1]][1])
                elif controller.selected_region >= 0:
                    camera.focus(scene.regions[controller.selected_region][1])
            elif key == ord("i"):
                guide.request(controller.selected_part, describe_part(controller.selected_part))
    finally:
        if hands is not None:
            hands.close()
        if capture is not None:
            capture.release()
        renderer.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())