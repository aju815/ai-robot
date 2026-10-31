# HOLO-HUMAN

This folder is a standalone Python hologram preview. It includes a smooth, articulated
procedural robot so you can run it without downloading a model. An optional
Mixamo-compatible humanoid `.glb` can be provided for a more detailed character.

The webcam tracks both hands: move each hand to control the corresponding robot arm,
and pinch/flex fingers to move the robot's hand joints. No camera image or face is
displayed. Pinch a robot part to select it and print its explanation in Terminal;
pinch with both hands and move them apart/together to zoom in/out. Finger-count
gestures follow the SAM-X1 interaction pattern: a fist enters standby, and 1-5 fingers
select head, torso, arms, legs, or full-body control. Point with the index finger and
pinch over a body region to select it directly. The robot is the only content
displayed in the preview.
When both hands are visible, each independently aims the matching robot arm; the
right hand (or left if right is not visible) selects the SAM finger-count mode.

Every articulated mesh can be selected by pointing at it and pinching. The selected
part's built-in description and optional Ollama explanation print in Terminal. Press
**I** to request a longer local Ollama explanation. Zoom is manual: use **+**/**-** or
the mouse wheel, **F** to focus the pointed/selected part, and **0** to restore the
original camera view.
Zoom changes only when you use a zoom gesture or key; it does not change on its own.

## Run

From the project root, install the robot's dependencies and run the built-in robot:

```bash
python3 -m pip install -r holo_robot/requirements.txt
python3 -B -m holo_robot
```

Press **Q** or **Esc** to exit; press **S** to save a screenshot. If the webcam cannot
be opened, select another device with `--camera 1`. To preview the avatar without a
webcam, use `--no-camera`.
The command above enables the webcam. If macOS cannot open it, run
`python3 -B -m holo_robot --list-cameras` and allow Camera access for Terminal or
Visual Studio Code under **System Settings > Privacy & Security > Camera**.
The robot is shown alone by default; add `--show-demo-objects` to display the floating
sample objects as well.
To use a downloaded humanoid model instead, pass its actual file path:

```bash
python3 -m holo_robot "$HOME/Downloads/actual-model-name.glb"
```

Additional Mixamo-compatible animation GLBs can be supplied with one or more
`--animation /path/to/animation.glb` options.

## Ollama part explanations

Install and start Ollama, then download a local model. For example:

```bash
ollama pull qwen2.5vl:7b
```

The app requests an explanation when you pinch-select a robot part. You can also press
**I** while a robot part is selected. If using another model or Ollama server URL,
set `HOLO_OLLAMA_MODEL` and `OLLAMA_URL`, or pass
`--ollama-model MODEL --ollama-url URL`. Explanations are generated in a background
thread so webcam tracking and rendering remain responsive. If Ollama is unavailable,
the built-in description still prints and the connection error is shown in Terminal.

The built-in humanoid is a procedural white-and-black mechanical robot with cyan
hologram lighting, not a photorealistic model. A more detailed character needs an
external model containing triangle meshes and a Mixamo-compatible skeleton with hips,
head, and left/right arm, forearm, and hand bones.

The current window title is **HOLO ROBOT 3D v9 | ROBOT ONLY**. The terminal
prints the exact launcher path before starting. Close any older robot window before
launching v9; Python processes already running do not reload changed files.
