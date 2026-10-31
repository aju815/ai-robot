"""Local Ollama explanations for selectable humanoid components."""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request


PART_DETAILS = {
    "HEAD UNIT": (
        "The head unit contains the visor, optical sensors, and neck actuator. "
        "It turns to follow a tracked hand and represents the robot's vision and "
        "orientation system."
    ),
    "TORSO CORE": (
        "The torso contains the chest armor, segmented abdomen, and glowing power "
        "core. The spine and chest joints lean together when you move your hand."
    ),
    "LEFT ARM": (
        "The left arm contains a shoulder actuator, upper-arm armor, an elbow servo, "
        "forearm, wrist, palm, and five articulated fingers. Your left hand controls "
        "its reach and finger curl."
    ),
    "RIGHT ARM": (
        "The right arm contains a shoulder actuator, upper-arm armor, an elbow servo, "
        "forearm, wrist, palm, and five articulated fingers. Your right hand controls "
        "its reach and finger curl."
    ),
    "LEFT LEG": (
        "The left leg contains a hip actuator, thigh armor, knee joint, shin, ankle, "
        "and foot. It responds to the leg-spread and crouch gestures."
    ),
    "RIGHT LEG": (
        "The right leg contains a hip actuator, thigh armor, knee joint, shin, ankle, "
        "and foot. It responds to the leg-spread and crouch gestures."
    ),
    "HOLOGRAM ROBOT": (
        "This articulated hologram combines a humanoid skeleton, separate armor "
        "components, hand tracking, and a projected scan-line display. Select a body "
        "region to inspect its individual parts."
    ),
}


PART_COMPONENT_DETAILS = {
    "OPTIC": "A glowing optical indicator on the helmet. It is a visual sensor detail in this simulation, not a working camera.",
    "VISOR": "The dark face visor is a display surface that gives the robot its face. It does not contain a functional screen.",
    "HELMET": "The helmet is the robot's smooth head shell. The head joint lets the whole assembly turn and nod.",
    "NECK JOINT": "The neck joint connects the head to the chest and lets the head rotate independently.",
    "CHEST POWER CORE": "The bright chest core is a simulated holographic status light, not a physical power source.",
    "CHEST CORE HOUSING": "This chest housing surrounds the simulated status light and is attached to the chest shell.",
    "CHEST SHELL": "The chest shell forms the upper torso armor and follows the spine and chest joints.",
    "ABDOMEN": "The abdomen armor is attached to the lower spine and bends when the torso leans.",
    "ABDOMINAL RIB": "These small plates represent segmented abdomen armor; they are visual details, not separate actuators.",
    "SHOULDER ACTUATOR": "The shoulder actuator is the pivot where the upper arm joins the chest; hand tracking rotates this joint.",
    "UPPER ARM ARMOR": "This shell covers the upper-arm segment between the shoulder and elbow.",
    "ELBOW JOINT": "The elbow joint connects the upper arm to the forearm and bends as the tracked hand moves.",
    "FOREARM ARMOR": "This shell covers the segment between the elbow and wrist.",
    "PALM": "The palm is the center of the robot hand. Its position follows the tracked wrist and hand.",
    "THUMB SEGMENT": "This thumb segment is articulated and bends with the tracked thumb curl.",
    "INDEX SEGMENT": "This index-finger segment is articulated and bends with the tracked index-finger curl.",
    "MIDDLE SEGMENT": "This middle-finger segment is articulated and bends with the tracked middle-finger curl.",
    "RING SEGMENT": "This ring-finger segment is articulated and bends with the tracked ring-finger curl.",
    "PINKY SEGMENT": "This pinky segment is articulated and bends with the tracked pinky curl.",
    "FINGER SEGMENT": "This small articulated finger segment bends with the corresponding tracked finger.",
    "HIP ACTUATOR": "The hip pivot connects the torso to the upper leg and participates in the spread and crouch poses.",
    "THIGH ARMOR": "The thigh shell covers the upper leg between the hip and knee.",
    "KNEE JOINT": "The knee joint connects the thigh and shin; this demo bends it during the crouch gesture.",
    "SHIN ARMOR": "The shin shell covers the lower-leg segment between the knee and ankle.",
    "ANKLE": "The ankle connects the lower leg to the foot and is a visual articulation point in this model.",
    "FOOT SHELL": "The foot shell forms the robot's base of support and stays attached to the lower leg.",
}


def describe_part(part: str) -> str:
    """Return a short deterministic description for a robot part."""
    for component, description in PART_COMPONENT_DETAILS.items():
        if component in part:
            return description
    for key, description in PART_DETAILS.items():
        if part == key or part.startswith(key + " /"):
            return description
    category = next(
        (key for key in ("HEAD UNIT", "TORSO CORE", "LEFT ARM", "RIGHT ARM",
                         "LEFT LEG", "RIGHT LEG") if key in part),
        "HOLOGRAM ROBOT",
    )
    return f"{part}: {PART_DETAILS[category]}"


class OllamaGuide:
    """Generate part explanations asynchronously so model startup never blocks rendering."""

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.text = ""
        self.status = "Press I for an Ollama explanation."
        self.part = ""
        self._lock = threading.Lock()
        self._busy = False

    def request(self, part: str, description: str) -> bool:
        with self._lock:
            if self._busy:
                self.status = "Ollama is still explaining the previous part."
                return False
            self._busy = True
            self.part = part
            self.text = ""
            self.status = f"Loading {self.model} for {part}..."
        threading.Thread(
            target=self._generate,
            args=(part, description),
            name="holo-ollama-guide",
            daemon=True,
        ).start()
        return True

    def _generate(self, part: str, description: str) -> None:
        prompt = (
            "You are a robotics educator explaining a stylized, simulated hologram robot. "
            "Do not claim these parts are real hardware. Explain only the specific named "
            "component, not the whole limb. Write one short paragraph followed by exactly "
            "three concise bullet points: purpose, movement in this demo, and what the "
            "user can observe. Do not invent sensors, power sources, or specifications "
            "that are not stated in the demo notes.\n\n"
            f"Part: {part}\nDemo notes: {description}"
        )
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0.3, "num_predict": 220},
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                result = json.load(response)
            answer = result.get("message", {}).get("content", "").strip()
            if not answer:
                raise ValueError("Ollama returned an empty explanation.")
        except urllib.error.HTTPError as error:
            message = error.read().decode("utf-8", errors="replace").strip()
            status = f"Ollama HTTP {error.code}: {message or error.reason}"
            answer = ""
        except urllib.error.URLError as error:
            status = f"Cannot reach Ollama at {self.base_url}: {error.reason}"
            answer = ""
        except TimeoutError:
            status = f"Ollama request timed out after 90 seconds ({self.base_url})."
            answer = ""
        except OSError as error:
            status = f"Ollama connection failed: {error}"
            answer = ""
        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as error:
            status = f"Invalid Ollama response: {error}"
            answer = ""
        with self._lock:
            self.text = answer
            self.status = "Ollama explanation ready." if answer else status
            self._busy = False
