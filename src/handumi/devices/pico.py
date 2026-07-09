"""PICO / XRoboToolkit tracking helpers for HandUMI recording."""

from __future__ import annotations

import hashlib
import logging
import socket
import subprocess
import time

import numpy as np

log = logging.getLogger("handumi.record")

SERVICE_SCRIPT = "/opt/apps/roboticsservice/runService.sh"
SERVICE_WAIT_S = 3.0

# Port on which RoboticsServiceProcess accepts connections from the PICO app.
# The PC service also exposes a local gRPC endpoint on 60061 for the Python SDK,
# but 63901 is the port the PICO VR app dials into.
PICO_SERVICE_PORT = 63901

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


def guess_lan_ip() -> str | None:
    """Best-effort local LAN IP hint for XRoboToolkit WiFi mode."""

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return None


def verify_adb_connection(timeout_s: float = 15.0) -> bool:
    """Return whether at least one ADB device is connected within ``timeout_s``."""
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
        devices = [ln for ln in lines[1:] if "\tdevice" in ln]
        if devices:
            log.info(f"ADB device(s) detected: {devices}")
            return True
        remaining = int(deadline - time.time())
        log.info(f"No ADB device found yet - retrying (timeout in {remaining}s) ...")
        time.sleep(1.0)
    return False


def setup_adb_reverse() -> None:
    """Set up ADB reverse port forwarding for PICO USB mode."""
    log.info(f"Setting up ADB reverse tunnel for PICO port {PICO_SERVICE_PORT} ...")
    try:
        result = subprocess.run(
            ["adb", "reverse", f"tcp:{PICO_SERVICE_PORT}", f"tcp:{PICO_SERVICE_PORT}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            log.info(
                f"ADB reverse tcp:{PICO_SERVICE_PORT} -> localhost:{PICO_SERVICE_PORT} OK. "
                "Set PC-service IP to 127.0.0.1 in the PICO app."
            )
        else:
            log.warning(
                f"adb reverse returned non-zero ({result.returncode}): {result.stderr.strip()}"
            )
    except FileNotFoundError:
        log.warning("'adb' not found - skipping reverse tunnel setup.")
    except subprocess.TimeoutExpired:
        log.warning("adb reverse timed out - skipping.")


def launch_xrt_service() -> None:
    """Start the XRoboToolkit PC service."""
    log.info(f"Launching XRoboToolkit PC service: {SERVICE_SCRIPT}")
    try:
        subprocess.Popen(["bash", SERVICE_SCRIPT])
    except FileNotFoundError:
        log.warning(
            f"Service script not found at {SERVICE_SCRIPT}. "
            "Assuming the service is already running."
        )
    log.info(f"Waiting {SERVICE_WAIT_S}s for service to initialise ...")
    time.sleep(SERVICE_WAIT_S)


def init_xrt():
    """Import and initialise xrobotoolkit_sdk. Returns the module."""
    try:
        import xrobotoolkit_sdk as xrt
    except ImportError as exc:
        raise SystemExit(
            f"ERROR: could not import xrobotoolkit_sdk: {exc}\n"
            "Run  bin/install.sh  to build/install the SDK."
        ) from exc
    log.info("Calling xrt.init() ...")
    xrt.init()
    log.info("xrobotoolkit_sdk initialised.")
    return xrt


def wait_for_pico_data(xrt, *, mode: str, timeout_s: float = 15.0) -> bool:
    """Block until the requested PICO data stream is available or timeout expires."""
    if mode == "whole-body":
        log.info("Waiting for PICO body-tracking data ...")
    elif mode == "object":
        log.info("Waiting for PICO motion tracker / object-tracking data ...")
    else:
        log.info("Waiting for PICO controller data ...")

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
        log.info("  ... still waiting for PICO data")
        time.sleep(1.0)


def safe_array(value, shape: tuple[int, ...], dtype=np.float32) -> np.ndarray:
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


def safe_call_array(fn, shape: tuple[int, ...], dtype=np.float32) -> np.ndarray:
    try:
        value = fn()
    except Exception:  # noqa: BLE001 - disconnected XR streams become zeros.
        return np.zeros(shape, dtype=dtype)
    return safe_array(value, shape, dtype=dtype)


def read_start_button_value(xrt, button: str) -> float:
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

    log.info(f"  Press PICO '{button}' to start recording ...")
    while not stop_event.is_set():
        if read_start_button_value(xrt, button) >= threshold:
            while (
                read_start_button_value(xrt, button) >= threshold
                and not stop_event.is_set()
            ):
                time.sleep(0.02)
            return not stop_event.is_set()
        time.sleep(0.02)
    return False


def wait_for_button_release(
    xrt,
    *,
    button: str,
    threshold: float,
    stop_event,
) -> None:
    while (
        read_start_button_value(xrt, button) >= threshold
        and not stop_event.is_set()
    ):
        time.sleep(0.02)


def wait_for_manual_start(
    xrt,
    *,
    start_button: str,
    finish_button: str,
    threshold: float,
    stop_event,
) -> str:
    log.info(f"  Press PICO '{start_button}' to start, '{finish_button}' to finish ...")
    prev_start = read_start_button_value(xrt, start_button) >= threshold
    prev_finish = read_start_button_value(xrt, finish_button) >= threshold
    while not stop_event.is_set():
        start_pressed = read_start_button_value(xrt, start_button) >= threshold
        finish_pressed = read_start_button_value(xrt, finish_button) >= threshold
        start_rise = start_pressed and not prev_start
        finish_rise = finish_pressed and not prev_finish
        prev_start, prev_finish = start_pressed, finish_pressed

        if finish_rise:
            wait_for_button_release(
                xrt,
                button=finish_button,
                threshold=threshold,
                stop_event=stop_event,
            )
            return "finish"
        if start_rise:
            wait_for_button_release(
                xrt,
                button=start_button,
                threshold=threshold,
                stop_event=stop_event,
            )
            return "start"
        time.sleep(0.02)
    return "finish"


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
    """Read one snapshot of all PICO sensors."""
    frame: dict = {
        "observation.pico.timestamp_ns": np.array(
            [xrt.get_time_stamp_ns()], dtype=np.int64
        ),
        "observation.pico.headset_pose": safe_call_array(
            xrt.get_headset_pose, (7,), dtype=np.float32
        ),
        "observation.pico.left_controller_pose": safe_call_array(
            xrt.get_left_controller_pose, (7,), dtype=np.float32
        ),
        "observation.pico.right_controller_pose": safe_call_array(
            xrt.get_right_controller_pose, (7,), dtype=np.float32
        ),
    }

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
