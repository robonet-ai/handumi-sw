#!/usr/bin/env python3
"""
Record data from SO100 leader arm, cameras, and PICO headset (via XRoboToolkit).

What gets recorded per frame
─────────────────────────────
  observation.images.cam_0/1/2      RGB frames from 3 OpenCV cameras  (video)
  observation.state                 6 SO100 leader motor positions     (float32[6])
  action                            same as observation.state          (float32[6])
  observation.pico.body_joints_pose       24 joint poses [x,y,z,qx,qy,qz,qw]   (float32[24,7])
  observation.pico.body_joints_velocity   24 joint velocities [vx,vy,vz,wx,wy,wz] (float32[24,6])
  observation.pico.body_joints_accel      24 joint accelerations                 (float32[24,6])
  observation.pico.left_controller_pose   [x,y,z,qx,qy,qz,qw]                  (float32[7])
  observation.pico.right_controller_pose  [x,y,z,qx,qy,qz,qw]                  (float32[7])
  observation.pico.left_hand_pose         27 hand joint poses                    (float32[27,7])
  observation.pico.right_hand_pose        27 hand joint poses                    (float32[27,7])
  observation.pico.timestamp_ns           PICO HW timestamp                      (int64[1])

No follower robot is required.  Action = leader positions, which can be used
directly to train an imitation-learning policy.

Usage
─────
  python test/read_pico_cameras_motors.py \
      --cam-ids 0 2 4 \
      --motor-port /dev/ttyUSB0 \
      --repo-id local/my_dataset \
      --output-dir datasets/my_dataset \
      --task "Pick and place cube" \
      --num-episodes 10 \
      --episode-time-s 60 \
      --fps 30

  (See bin/record.sh for a ready-made launcher.)
"""

import argparse
import hashlib
import logging
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s – %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dexumi.record")

# ── Constants ──────────────────────────────────────────────────────────────────
SERVICE_SCRIPT = "/opt/apps/roboticsservice/runService.sh"
SERVICE_WAIT_S = 3.0

# Port on which RoboticsServiceProcess accepts connections from the PICO app.
# The PC service also exposes a local gRPC endpoint on 60061 for the Python SDK,
# but 63901 is the port the PICO VR app dials into.
PICO_SERVICE_PORT = 63901

MOTOR_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

MAX_MOTION_TRACKERS = 2
START_BUTTON_CHOICES = [
    "enter",
    "A",
    "B",
    "X",
    "Y",
    "left_menu",
    "right_menu",
    "left_trigger",
    "right_trigger",
    "left_grip",
    "right_grip",
]

# ── PICO / XRoboToolkit helpers ────────────────────────────────────────────────


def guess_lan_ip() -> str | None:
    """Best-effort local LAN IP hint for XRoboToolkit WiFi mode."""

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return None


def verify_adb_connection(timeout_s: float = 15.0) -> bool:
    """
    Check that at least one ADB device is connected.
    Polls every second for up to timeout_s.
    Returns True if a device is found; False on timeout.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            result = subprocess.run(
                ["adb", "devices"], capture_output=True, text=True, timeout=5
            )
        except FileNotFoundError:
            raise SystemExit(
                "ERROR: 'adb' not found in PATH.\n"
                "Install Android Debug Bridge: sudo apt install adb"
            )
        lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
        # lines[0] is always "List of devices attached"
        devices = [ln for ln in lines[1:] if "\tdevice" in ln]
        if devices:
            log.info(f"ADB device(s) detected: {devices}")
            return True
        remaining = int(deadline - time.time())
        log.info(f"No ADB device found yet – retrying (timeout in {remaining}s) …")
        time.sleep(1.0)
    return False


def setup_adb_reverse() -> None:
    """Set up ADB reverse port forwarding so the PICO app can reach the PC service
    over USB instead of WiFi.

    After this call the PICO app should use 127.0.0.1 as the PC-service IP.
    """
    log.info(f"Setting up ADB reverse tunnel for PICO port {PICO_SERVICE_PORT} …")
    try:
        result = subprocess.run(
            ["adb", "reverse", f"tcp:{PICO_SERVICE_PORT}", f"tcp:{PICO_SERVICE_PORT}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            log.info(
                f"ADB reverse tcp:{PICO_SERVICE_PORT} → localhost:{PICO_SERVICE_PORT} OK. "
                "Set PC-service IP to 127.0.0.1 in the PICO app."
            )
        else:
            log.warning(
                f"adb reverse returned non-zero ({result.returncode}): {result.stderr.strip()}"
            )
    except FileNotFoundError:
        log.warning("'adb' not found – skipping reverse tunnel setup.")
    except subprocess.TimeoutExpired:
        log.warning("adb reverse timed out – skipping.")


def launch_xrt_service() -> None:
    """Start the XRoboToolkit PC service (idempotent – ignores 'already running')."""
    log.info(f"Launching XRoboToolkit PC service: {SERVICE_SCRIPT}")
    try:
        subprocess.Popen(["bash", SERVICE_SCRIPT])
    except FileNotFoundError:
        log.warning(
            f"Service script not found at {SERVICE_SCRIPT}. "
            "Assuming the service is already running."
        )
    log.info(f"Waiting {SERVICE_WAIT_S}s for service to initialise …")
    time.sleep(SERVICE_WAIT_S)


def init_xrt():
    """Import and initialise xrobotoolkit_sdk.  Returns the module."""
    try:
        import xrobotoolkit_sdk as xrt
    except ImportError as exc:
        raise SystemExit(
            f"ERROR: could not import xrobotoolkit_sdk: {exc}\n"
            "Run  bin/install.sh  to build/install the SDK."
        ) from exc
    log.info("Calling xrt.init() …")
    xrt.init()
    log.info("xrobotoolkit_sdk initialised.")
    return xrt


def wait_for_pico_data(xrt, *, mode: str, timeout_s: float = 15.0) -> bool:
    """Block until the requested PICO data stream is available or timeout expires."""
    if mode == "whole-body":
        log.info("Waiting for PICO body-tracking data …")
    elif mode == "object":
        log.info("Waiting for PICO motion tracker / object-tracking data …")
    else:
        log.info("Waiting for PICO controller data …")

    deadline = time.time() + timeout_s
    while True:
        if mode == "whole-body" and xrt.is_body_data_available():
            log.info("PICO body-tracking data is available.")
            return True
        if mode == "object":
            try:
                if xrt.num_motion_data_available() > 0:
                    log.info("PICO motion tracker data is available.")
                    return True
            except AttributeError:
                log.warning("xrobotoolkit_sdk does not expose motion tracker APIs.")
                return False
        if mode == "mandos":
            try:
                np.asarray(xrt.get_left_controller_pose(), dtype=np.float32)
                np.asarray(xrt.get_right_controller_pose(), dtype=np.float32)
                log.info("PICO controller data is available.")
                return True
            except Exception:  # noqa: BLE001 - SDK may raise while app connects.
                pass

        if time.time() > deadline:
            return False
        log.info("  … still waiting for PICO data")
        time.sleep(1.0)


def _safe_array(value, shape: tuple[int, ...], dtype=np.float32) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=dtype)
    except Exception:  # noqa: BLE001 - SDK failures should not break fixed schema.
        return np.zeros(shape, dtype=dtype)
    if arr.shape != shape:
        out = np.zeros(shape, dtype=dtype)
        slices = tuple(slice(0, min(a, b)) for a, b in zip(arr.shape, shape, strict=False))
        try:
            out[slices] = arr[slices]
        except Exception:  # noqa: BLE001
            return np.zeros(shape, dtype=dtype)
        return out
    return arr


def _safe_call_array(fn, shape: tuple[int, ...], dtype=np.float32) -> np.ndarray:
    try:
        value = fn()
    except Exception:  # noqa: BLE001 - disconnected XR streams become zeros.
        return np.zeros(shape, dtype=dtype)
    return _safe_array(value, shape, dtype=dtype)


def _read_start_button_value(xrt, button: str) -> float:
    """Return normalized pressed value for the configured start button."""

    readers = {
        "A": xrt.get_A_button,
        "B": xrt.get_B_button,
        "X": xrt.get_X_button,
        "Y": xrt.get_Y_button,
        "left_menu": xrt.get_left_menu_button,
        "right_menu": xrt.get_right_menu_button,
        "left_trigger": xrt.get_left_trigger,
        "right_trigger": xrt.get_right_trigger,
        "left_grip": xrt.get_left_grip,
        "right_grip": xrt.get_right_grip,
    }
    try:
        return float(readers[button]())
    except Exception:  # noqa: BLE001 - XR stream may still be settling.
        return 0.0


def wait_for_start_button(
    xrt,
    *,
    button: str,
    threshold: float,
    stop_event,
) -> bool:
    """Wait until the configured controller input starts an episode."""

    log.info(f"  Press PICO '{button}' to start recording …")
    while not stop_event.is_set():
        if _read_start_button_value(xrt, button) >= threshold:
            # Start after release so frame 0 is motion, not the trigger press.
            while (
                _read_start_button_value(xrt, button) >= threshold
                and not stop_event.is_set()
            ):
                time.sleep(0.02)
            return not stop_event.is_set()
        time.sleep(0.02)
    return False


def _serial_hash(serial: object) -> np.int64:
    digest = hashlib.blake2b(str(serial).encode("utf-8"), digest_size=8).digest()
    return np.int64(int.from_bytes(digest, "little", signed=False) & ((1 << 63) - 1))


def read_motion_trackers(
    xrt,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    poses = np.zeros((MAX_MOTION_TRACKERS, 7), dtype=np.float32)
    velocities = np.zeros((MAX_MOTION_TRACKERS, 6), dtype=np.float32)
    accelerations = np.zeros((MAX_MOTION_TRACKERS, 6), dtype=np.float32)
    serial_hashes = np.zeros((MAX_MOTION_TRACKERS,), dtype=np.int64)
    count = np.zeros((1,), dtype=np.int64)

    try:
        n_available = int(xrt.num_motion_data_available())
    except AttributeError:
        log.warning("xrobotoolkit_sdk does not expose motion tracker APIs.")
        return poses, velocities, accelerations, count, serial_hashes
    except Exception as exc:  # noqa: BLE001
        log.warning(f"Could not read motion tracker count: {exc}")
        return poses, velocities, accelerations, count, serial_hashes

    n = min(n_available, MAX_MOTION_TRACKERS)
    count[0] = n_available
    if n == 0:
        return poses, velocities, accelerations, count, serial_hashes

    try:
        raw_poses = np.atleast_2d(
            np.asarray(xrt.get_motion_tracker_pose(), dtype=np.float32)
        )
        raw_velocities = np.atleast_2d(
            np.asarray(xrt.get_motion_tracker_velocity(), dtype=np.float32)
        )
        raw_accelerations = np.atleast_2d(
            np.asarray(xrt.get_motion_tracker_acceleration(), dtype=np.float32)
        )
        raw_serials = list(xrt.get_motion_tracker_serial_numbers())
    except Exception as exc:  # noqa: BLE001
        log.warning(f"Could not read motion tracker data: {exc}")
        return poses, velocities, accelerations, count, serial_hashes

    poses[:n] = raw_poses[:n, :7]
    velocities[:n] = raw_velocities[:n, :6]
    accelerations[:n] = raw_accelerations[:n, :6]
    for i, serial in enumerate(raw_serials[:n]):
        serial_hashes[i] = _serial_hash(serial)
    if raw_serials:
        log.debug(f"Motion trackers: {raw_serials[:n]}")
    return poses, velocities, accelerations, count, serial_hashes


def empty_pico_frame() -> dict:
    """Return a complete zero-filled PICO frame with the fixed dataset schema."""

    return {
        "observation.pico.timestamp_ns": np.zeros((1,), dtype=np.int64),
        "observation.pico.headset_pose": np.zeros((7,), dtype=np.float32),
        "observation.pico.left_controller_pose": np.zeros((7,), dtype=np.float32),
        "observation.pico.right_controller_pose": np.zeros((7,), dtype=np.float32),
        "observation.pico.body_joints_pose": np.zeros((24, 7), dtype=np.float32),
        "observation.pico.body_joints_velocity": np.zeros((24, 6), dtype=np.float32),
        "observation.pico.body_joints_accel": np.zeros((24, 6), dtype=np.float32),
        "observation.pico.left_hand_pose": np.zeros((27, 7), dtype=np.float32),
        "observation.pico.right_hand_pose": np.zeros((27, 7), dtype=np.float32),
        "observation.pico.motion_tracker_pose": np.zeros(
            (MAX_MOTION_TRACKERS, 7), dtype=np.float32
        ),
        "observation.pico.motion_tracker_velocity": np.zeros(
            (MAX_MOTION_TRACKERS, 6), dtype=np.float32
        ),
        "observation.pico.motion_tracker_accel": np.zeros(
            (MAX_MOTION_TRACKERS, 6), dtype=np.float32
        ),
        "observation.pico.motion_tracker_count": np.zeros((1,), dtype=np.int64),
        "observation.pico.motion_tracker_serial_hash": np.zeros(
            (MAX_MOTION_TRACKERS,), dtype=np.int64
        ),
    }


def read_pico_frame(xrt, *, mode: str) -> dict:
    """
    Read one snapshot of all PICO sensors.

    Missing / inactive channels are filled with zeros so that the dataset
    schema stays fixed across every frame.
    """
    frame: dict = {
        "observation.pico.timestamp_ns": np.array(
            [xrt.get_time_stamp_ns()], dtype=np.int64
        ),
        "observation.pico.headset_pose": _safe_call_array(
            xrt.get_headset_pose, (7,), dtype=np.float32
        ),
        "observation.pico.left_controller_pose": _safe_call_array(
            xrt.get_left_controller_pose, (7,), dtype=np.float32
        ),
        "observation.pico.right_controller_pose": _safe_call_array(
            xrt.get_right_controller_pose, (7,), dtype=np.float32
        ),
    }

    # Body tracking
    if mode == "whole-body" and xrt.is_body_data_available():
        frame["observation.pico.body_joints_pose"] = np.array(
            xrt.get_body_joints_pose(), dtype=np.float32
        )
        frame["observation.pico.body_joints_velocity"] = np.array(
            xrt.get_body_joints_velocity(), dtype=np.float32
        )
        frame["observation.pico.body_joints_accel"] = np.array(
            xrt.get_body_joints_acceleration(), dtype=np.float32
        )
    else:
        frame["observation.pico.body_joints_pose"] = np.zeros((24, 7), dtype=np.float32)
        frame["observation.pico.body_joints_velocity"] = np.zeros((24, 6), dtype=np.float32)
        frame["observation.pico.body_joints_accel"] = np.zeros((24, 6), dtype=np.float32)

    # Hand tracking (fill with zeros when inactive)
    lh = (
        np.array(xrt.get_left_hand_tracking_state(), dtype=np.float32)
        if xrt.get_left_hand_is_active()
        else np.zeros((27, 7), dtype=np.float32)
    )
    rh = (
        np.array(xrt.get_right_hand_tracking_state(), dtype=np.float32)
        if xrt.get_right_hand_is_active()
        else np.zeros((27, 7), dtype=np.float32)
    )
    frame["observation.pico.left_hand_pose"] = lh
    frame["observation.pico.right_hand_pose"] = rh

    tracker_pose, tracker_vel, tracker_accel, tracker_count, tracker_serial_hashes = (
        read_motion_trackers(xrt) if mode == "object" else (
            np.zeros((MAX_MOTION_TRACKERS, 7), dtype=np.float32),
            np.zeros((MAX_MOTION_TRACKERS, 6), dtype=np.float32),
            np.zeros((MAX_MOTION_TRACKERS, 6), dtype=np.float32),
            np.zeros((1,), dtype=np.int64),
            np.zeros((MAX_MOTION_TRACKERS,), dtype=np.int64),
        )
    )
    frame["observation.pico.motion_tracker_pose"] = tracker_pose
    frame["observation.pico.motion_tracker_velocity"] = tracker_vel
    frame["observation.pico.motion_tracker_accel"] = tracker_accel
    frame["observation.pico.motion_tracker_count"] = tracker_count
    frame["observation.pico.motion_tracker_serial_hash"] = tracker_serial_hashes

    return frame


# ── Dataset feature schema ─────────────────────────────────────────────────────


def build_features(
    cam_names: list[str],
    cam_width: int,
    cam_height: int,
    use_videos: bool,
) -> dict:
    """
    Build the feature schema dict expected by LeRobotDataset.create().
    """
    img_dtype = "video" if use_videos else "image"
    features: dict = {}

    for cam in cam_names:
        features[f"observation.images.{cam}"] = {
            "dtype": img_dtype,
            "shape": (cam_height, cam_width, 3),
            "names": ["height", "width", "channel"],
        }

    features["observation.state"] = {
        "dtype": "float32",
        "shape": (len(MOTOR_NAMES),),
        "names": MOTOR_NAMES,
    }
    features["action"] = {
        "dtype": "float32",
        "shape": (len(MOTOR_NAMES),),
        "names": MOTOR_NAMES,
    }

    # PICO body tracking
    features["observation.pico.body_joints_pose"] = {
        "dtype": "float32",
        "shape": (24, 7),
        "names": None,
    }
    features["observation.pico.body_joints_velocity"] = {
        "dtype": "float32",
        "shape": (24, 6),
        "names": None,
    }
    features["observation.pico.body_joints_accel"] = {
        "dtype": "float32",
        "shape": (24, 6),
        "names": None,
    }

    # PICO controller / hand poses
    features["observation.pico.left_controller_pose"] = {
        "dtype": "float32",
        "shape": (7,),
        "names": ["x", "y", "z", "qx", "qy", "qz", "qw"],
    }
    features["observation.pico.headset_pose"] = {
        "dtype": "float32",
        "shape": (7,),
        "names": ["x", "y", "z", "qx", "qy", "qz", "qw"],
    }
    features["observation.pico.right_controller_pose"] = {
        "dtype": "float32",
        "shape": (7,),
        "names": ["x", "y", "z", "qx", "qy", "qz", "qw"],
    }
    features["observation.pico.left_hand_pose"] = {
        "dtype": "float32",
        "shape": (27, 7),
        "names": None,
    }
    features["observation.pico.right_hand_pose"] = {
        "dtype": "float32",
        "shape": (27, 7),
        "names": None,
    }
    features["observation.pico.timestamp_ns"] = {
        "dtype": "int64",
        "shape": (1,),
        "names": None,
    }
    features["observation.pico.motion_tracker_pose"] = {
        "dtype": "float32",
        "shape": (MAX_MOTION_TRACKERS, 7),
        "names": None,
    }
    features["observation.pico.motion_tracker_velocity"] = {
        "dtype": "float32",
        "shape": (MAX_MOTION_TRACKERS, 6),
        "names": None,
    }
    features["observation.pico.motion_tracker_accel"] = {
        "dtype": "float32",
        "shape": (MAX_MOTION_TRACKERS, 6),
        "names": None,
    }
    features["observation.pico.motion_tracker_count"] = {
        "dtype": "int64",
        "shape": (1,),
        "names": None,
    }
    features["observation.pico.motion_tracker_serial_hash"] = {
        "dtype": "int64",
        "shape": (MAX_MOTION_TRACKERS,),
        "names": None,
    }

    return features


# ── Recording loop ─────────────────────────────────────────────────────────────


def record_episode(
    *,
    dataset,
    cameras: list,
    cam_names: list[str],
    leader,
    xrt,
    episode_time_s: float,
    fps: int,
    task: str,
    stop_event,
    only_pico: bool,
    pico_mode: str,
    include_pico: bool,
    cam_width: int,
    cam_height: int,
) -> int:
    """
    Record one episode.  Returns the number of frames saved.
    stop_event is a threading.Event (or any object with .is_set()) that
    signals early termination.
    """
    control_interval = 1.0 / fps
    n_frames = 0
    start_t = time.perf_counter()

    while True:
        loop_start = time.perf_counter()

        elapsed = loop_start - start_t
        if elapsed >= episode_time_s or stop_event.is_set():
            break

        # ── Camera frames ──────────────────────────────────────────────────
        cam_frames: dict = {}
        for cam, name in zip(cameras, cam_names):
            frame = (
                np.zeros((cam_height, cam_width, 3), dtype=np.uint8)
                if only_pico
                else cam.async_read()
            )
            cam_frames[f"observation.images.{name}"] = frame  # HxWx3 uint8

        # ── Motor positions ────────────────────────────────────────────────
        if only_pico:
            state_vec = np.zeros((len(MOTOR_NAMES),), dtype=np.float32)
        else:
            motor_action: dict = leader.get_action()
            # get_action() → {"shoulder_pan.pos": float, …}
            state_vec = np.array(
                [motor_action[k] for k in MOTOR_NAMES], dtype=np.float32
            )

        # ── PICO data ──────────────────────────────────────────────────────
        if not include_pico:
            pico_frame = {}
        else:
            pico_frame = (
                read_pico_frame(xrt, mode=pico_mode)
                if xrt is not None
                else empty_pico_frame()
            )

        # ── Assemble and save frame ────────────────────────────────────────
        data_frame = {
            **cam_frames,
            "observation.state": state_vec,
            "action": state_vec.copy(),
            **pico_frame,
            "task": task,
        }
        dataset.add_frame(data_frame)
        n_frames += 1

        # ── Timing ────────────────────────────────────────────────────────
        dt = time.perf_counter() - loop_start
        sleep = control_interval - dt
        if sleep < 0:
            log.warning(
                f"Loop running slower than {fps} Hz "
                f"({1/dt:.1f} Hz actual). Consider reducing fps or camera resolution."
            )
        else:
            time.sleep(sleep)

    return n_frames


# ── Main ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Record SO100 leader + cameras + PICO into a LeRobot dataset."
    )

    # Cameras
    p.add_argument(
        "--cam-ids",
        nargs="+",
        type=int,
        default=[0, 2, 4],
        metavar="ID",
        help="OpenCV camera indices (3 expected, default: 0 2 4).",
    )
    p.add_argument("--cam-width", type=int, default=640)
    p.add_argument("--cam-height", type=int, default=480)
    p.add_argument("--cam-fps", type=int, default=30, help="Camera capture FPS.")

    # SO100 leader
    p.add_argument(
        "--motor-port",
        type=str,
        default="/dev/ttyUSB0",
        help="Serial port for the SO100 leader arm.",
    )
    p.add_argument(
        "--motor-id",
        type=str,
        default="leader",
        help="Identifier for the SO100 leader (used to locate calibration file).",
    )

    # Dataset
    p.add_argument(
        "--repo-id",
        type=str,
        default="local/dexumi_dataset",
        help="Dataset repo-id in '{namespace}/{name}' format.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Local root directory for the dataset (default: $HF_LEROBOT_HOME/<repo-id>).",
    )
    p.add_argument(
        "--task",
        type=str,
        default="Teleoperation recording",
        help="Short task description stored in every frame.",
    )
    p.add_argument("--num-episodes", type=int, default=10)
    p.add_argument(
        "--episode-time-s",
        type=float,
        default=60.0,
        help="Duration of each recording episode in seconds.",
    )
    p.add_argument("--fps", type=int, default=30, help="Dataset recording FPS.")
    p.add_argument(
        "--start-button",
        choices=START_BUTTON_CHOICES,
        default="enter",
        help=(
            "How to start each episode. Use 'enter' for keyboard, or a PICO "
            "controller input such as A, B, X, Y, left_trigger, or right_trigger."
        ),
    )
    p.add_argument(
        "--start-threshold",
        type=float,
        default=0.75,
        help="Analog threshold for trigger/grip start buttons.",
    )
    p.add_argument(
        "--no-video",
        action="store_true",
        help="Store camera frames as PNG images instead of video.",
    )
    p.add_argument(
        "--vcodec",
        type=str,
        default="h264",
        help="Video codec for encoding (h264, hevc, libsvtav1, auto).",
    )
    p.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the completed dataset to the Hugging Face Hub.",
    )
    p.add_argument(
        "--skip-pico",
        action="store_true",
        help="Disable PICO / XRoboToolkit (useful for dry-run without headset).",
    )
    p.add_argument(
        "--only-pico",
        action="store_true",
        help=(
            "Record only PICO/XR data. Camera images, observation.state, and action "
            "stay in the schema but are filled with zeros."
        ),
    )
    pico_mode = p.add_mutually_exclusive_group()
    pico_mode.add_argument(
        "--pico-mandos",
        action="store_true",
        help=(
            "Record PICO headset/controllers/hands only; body joints and motion "
            "trackers are zero-filled. Use this for controller-based arm inference."
        ),
    )
    pico_mode.add_argument(
        "--pico-object",
        action="store_true",
        help=(
            "Record PICO motion tracker/object-tracking poses. Body joints are "
            "zero-filled; headset/controller/hand poses are still recorded."
        ),
    )
    pico_mode.add_argument(
        "--pico-whole-body",
        action="store_true",
        help="Record the original PICO 24-joint body-tracking stream.",
    )
    pico_transport = p.add_mutually_exclusive_group()
    pico_transport.add_argument(
        "--pico-adb",
        action="store_true",
        help=(
            "Connect PICO through USB/ADB reverse tunnel. This is the default; "
            "set PC-service IP to 127.0.0.1 in the PICO app."
        ),
    )
    pico_transport.add_argument(
        "--pico-wifi",
        action="store_true",
        help=(
            "Connect PICO over WiFi/LAN. Skips ADB checks and adb reverse; set "
            "the PC-service IP in the PICO app to this computer's LAN IP."
        ),
    )
    p.add_argument(
        "--skip-adb-check",
        action="store_true",
        help="ADB mode only: skip the ADB device presence check.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.skip_pico and args.only_pico:
        raise SystemExit("--only-pico requires PICO; do not combine it with --skip-pico.")

    if args.pico_object:
        pico_mode = "object"
    elif args.pico_mandos:
        pico_mode = "mandos"
    else:
        pico_mode = "whole-body"
    pico_transport = "wifi" if args.pico_wifi else "adb"

    if args.only_pico:
        log.info(
            f"--only-pico set: cameras and SO100 leader will be zero-filled; "
            f"PICO mode={pico_mode!r}, transport={pico_transport!r}."
        )

    # ── 1. PICO / XRoboToolkit initialisation ─────────────────────────────────
    xrt = None
    if not args.skip_pico:
        log.info("─── PICO / XRoboToolkit setup ─────────────────────────")
        if pico_transport == "adb" and not args.skip_adb_check:
            log.info("Checking ADB connection …")
            if not verify_adb_connection(timeout_s=15.0):
                log.error(
                    "No ADB device found after 15 s.\n"
                    "  • Make sure the PICO headset is connected via USB cable.\n"
                    "  • Enable USB debugging on the headset.\n"
                    "  • Or pass --pico-wifi to connect through LAN/WiFi.\n"
                    "  • Or pass --skip-pico to record without the headset."
                )
                sys.exit(1)
            setup_adb_reverse()
        elif pico_transport == "wifi":
            lan_ip = guess_lan_ip()
            if lan_ip:
                log.info(
                    "PICO WiFi mode: skipping ADB. In the PICO XRoboToolkit app, "
                    f"set PC-service IP to {lan_ip} and port {PICO_SERVICE_PORT}."
                )
            else:
                log.info(
                    "PICO WiFi mode: skipping ADB. Set the PICO XRoboToolkit "
                    f"PC-service IP to this computer's LAN IP and port {PICO_SERVICE_PORT}."
                )
        else:
            log.info("ADB check skipped. Assuming XRoboToolkit can reach the PC service.")

        launch_xrt_service()
        xrt = init_xrt()

        if not wait_for_pico_data(xrt, mode=pico_mode, timeout_s=15.0):
            log.warning(
                f"PICO {pico_mode!r} data not available after 15 s. "
                "Missing streams will be filled with zeros."
            )
    else:
        log.info("--skip-pico set: PICO / XRoboToolkit disabled.")

    # ── 2. Camera initialisation ───────────────────────────────────────────────
    log.info("─── Camera setup ───────────────────────────────────────")
    cam_names = [f"cam_{i}" for i in range(len(args.cam_ids))]
    cameras: list = []
    if args.only_pico:
        cameras = [None for _ in cam_names]
        log.info(f"Only-PICO mode: {len(cam_names)} camera stream(s) will be zero-filled.")
    else:
        from lerobot.cameras.opencv import OpenCVCamera
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

        for cam_id, name in zip(args.cam_ids, cam_names):
            cfg = OpenCVCameraConfig(
                index_or_path=cam_id,
                fps=args.cam_fps,
                width=args.cam_width,
                height=args.cam_height,
            )
            cam = OpenCVCamera(cfg)
            cam.connect()
            cameras.append(cam)
            log.info(f"Camera '{name}' (index {cam_id}) connected.")

    # ── 3. SO100 leader initialisation ────────────────────────────────────────
    log.info("─── SO100 leader setup ─────────────────────────────────")
    leader = None
    if args.only_pico:
        log.info("Only-PICO mode: observation.state and action will be zero-filled.")
    else:
        from lerobot.teleoperators.so_leader import SO100Leader
        from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig

        leader_cfg = SOLeaderTeleopConfig(
            port=args.motor_port,
            id=args.motor_id,
            use_degrees=True,
        )
        leader = SO100Leader(leader_cfg)
        leader.connect()
        log.info(f"SO100 leader connected on {args.motor_port}.")

    # ── 4. Dataset creation ────────────────────────────────────────────────────
    log.info("─── Dataset setup ──────────────────────────────────────")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    use_videos = not args.no_video
    features = build_features(
        cam_names=cam_names,
        cam_width=args.cam_width,
        cam_height=args.cam_height,
        use_videos=use_videos,
    )

    # If PICO is disabled, remove PICO features so the schema is consistent
    if args.skip_pico:
        features = {k: v for k, v in features.items() if not k.startswith("observation.pico")}

    n_cams = len(cam_names)
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        root=args.output_dir,
        robot_type="so100_leader",
        features=features,
        use_videos=use_videos,
        image_writer_processes=0,
        image_writer_threads=max(1, 4 * n_cams),
        vcodec=args.vcodec,
    )
    log.info(f"Dataset created at: {dataset.root}")
    log.info(f"Features: {list(features.keys())}")

    # ── 5. Keyboard / signal stop event ───────────────────────────────────────
    import threading

    stop_event = threading.Event()

    def _on_signal(signum, frame):
        log.info("Signal received – stopping after current episode …")
        stop_event.set()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    # ── 6. Recording loop ──────────────────────────────────────────────────────
    log.info("─── Recording ──────────────────────────────────────────")
    log.info(f"Will record {args.num_episodes} episode(s) × {args.episode_time_s}s @ {args.fps} Hz.")
    log.info("Press Ctrl+C to stop early (current episode will still be saved).")

    recorded = 0
    try:
        while recorded < args.num_episodes and not stop_event.is_set():
            ep_num = dataset.num_episodes + 1
            log.info(f"\n── Episode {ep_num}/{args.num_episodes} ─────────────────────────────────")
            if args.start_button == "enter":
                input(f"  Press ENTER to start recording episode {ep_num} …")
            else:
                if xrt is None:
                    raise RuntimeError(
                        "--start-button requires PICO/XRoboToolkit; use 'enter' with --skip-pico."
                    )
                if not wait_for_start_button(
                    xrt,
                    button=args.start_button,
                    threshold=args.start_threshold,
                    stop_event=stop_event,
                ):
                    break

            log.info(f"  Recording for {args.episode_time_s}s …  (Ctrl+C to end early)")
            stop_event.clear()  # allow Ctrl+C to end only this episode
            n_frames = record_episode(
                dataset=dataset,
                cameras=cameras,
                cam_names=cam_names,
                leader=leader,
                xrt=xrt,
                episode_time_s=args.episode_time_s,
                fps=args.fps,
                task=args.task,
                stop_event=stop_event,
                only_pico=args.only_pico,
                pico_mode=pico_mode,
                include_pico=not args.skip_pico,
                cam_width=args.cam_width,
                cam_height=args.cam_height,
            )

            if n_frames == 0:
                log.warning("  No frames recorded – discarding episode.")
                dataset.clear_episode_buffer()
                continue

            log.info(f"  Saving {n_frames} frames …")
            dataset.save_episode()
            recorded += 1
            log.info(f"  Episode {ep_num} saved. ({recorded}/{args.num_episodes} done)")

            # After Ctrl+C ends the episode, ask whether to continue
            if stop_event.is_set() and recorded < args.num_episodes:
                ans = input("\nContinue recording more episodes? [y/N] ").strip().lower()
                if ans == "y":
                    stop_event.clear()
                else:
                    break

    finally:
        log.info("─── Finalising ─────────────────────────────────────────")
        dataset.finalize()

        for cam in cameras:
            try:
                cam.disconnect()
            except Exception:
                pass

        if leader is not None and leader.is_connected:
            leader.disconnect()

        if xrt is not None:
            try:
                xrt.close()
            except Exception:
                pass

        log.info(f"Done. Recorded {recorded} episode(s).")
        log.info(f"Dataset stored at: {dataset.root}")

        if args.push_to_hub:
            log.info("Pushing dataset to Hugging Face Hub …")
            dataset.push_to_hub()


if __name__ == "__main__":
    main()
