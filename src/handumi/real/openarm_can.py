"""Optional OpenArm v1 backend using the official ``openarm_can`` bindings."""

from __future__ import annotations

import importlib
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Protocol

import numpy as np
import yaml

log = logging.getLogger(__name__)

SIDES: tuple[str, str] = ("left", "right")
ARM_DOF = 7
SEND_CAN_IDS = tuple(range(0x01, 0x08))
RECV_CAN_IDS = tuple(range(0x11, 0x18))
GRIPPER_SEND_CAN_ID = 0x08
GRIPPER_RECV_CAN_ID = 0x18
DEFAULT_KP = (70.0, 70.0, 70.0, 60.0, 10.0, 10.0, 10.0)
DEFAULT_KD = (2.75, 2.5, 2.0, 2.0, 0.7, 0.6, 0.5)
JOINT_LIMIT_SNAP_TOLERANCE_RAD = 1e-4


@dataclass(frozen=True)
class OpenArmCanSettings:
    left_port: str = "can1"
    right_port: str = "can0"
    enable_fd: bool = True
    bitrate: int = 1_000_000
    dbitrate: int = 5_000_000
    command_rate_hz: float = 100.0
    max_joint_speed_rad_s: float = 1.0
    home_max_joint_speed_rad_s: float = 0.25
    home_timeout_s: float = 30.0
    home_tolerance_rad: float = 0.05
    watchdog_timeout_s: float = 0.15
    following_error_rad: float = 0.35
    gripper_closed_position_rad: float = 0.0
    gripper_open_position_rad: float = -1.0471975511965976
    left_gripper_closed_position_rad: float | None = None
    left_gripper_open_position_rad: float | None = None
    right_gripper_closed_position_rad: float | None = None
    right_gripper_open_position_rad: float | None = None
    kp: tuple[float, ...] = DEFAULT_KP
    kd: tuple[float, ...] = DEFAULT_KD


def load_openarm_settings(
    rig_config: Path,
    robot_real: dict[str, Any] | None = None,
    gripper_calibration_path: Path | None = None,
) -> OpenArmCanSettings:
    """Combine portable robot defaults with machine-local CAN assignments."""
    robot_real = robot_real or {}
    data: dict[str, Any] = {}
    if rig_config.exists():
        with rig_config.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    can = ((data.get("robots") or {}).get("openarmv1") or {}).get("can") or {}
    control = robot_real.get("control") or {}
    gains = robot_real.get("gains") or {}
    gripper = robot_real.get("gripper") or {}
    calibrated: dict[str, Any] = {}
    if gripper_calibration_path is not None and gripper_calibration_path.exists():
        with gripper_calibration_path.open("r", encoding="utf-8") as handle:
            calibrated = yaml.safe_load(handle) or {}

    def calibrated_value(side: str, key: str) -> float | None:
        value = (calibrated.get(side) or {}).get(key)
        return None if value is None else float(value)

    return OpenArmCanSettings(
        left_port=str(can.get("left_port", "can1")),
        right_port=str(can.get("right_port", "can0")),
        enable_fd=bool(can.get("fd", True)),
        bitrate=int(can.get("bitrate", 1_000_000)),
        dbitrate=int(can.get("dbitrate", 5_000_000)),
        command_rate_hz=float(control.get("command_rate_hz", 100.0)),
        max_joint_speed_rad_s=float(control.get("max_joint_speed_rad_s", 1.0)),
        home_max_joint_speed_rad_s=float(
            control.get("home_max_joint_speed_rad_s", 0.25)
        ),
        home_timeout_s=float(control.get("home_timeout_s", 30.0)),
        home_tolerance_rad=float(control.get("home_tolerance_rad", 0.05)),
        watchdog_timeout_s=float(control.get("watchdog_timeout_s", 0.15)),
        following_error_rad=float(control.get("following_error_rad", 0.35)),
        gripper_closed_position_rad=float(
            gripper.get("closed_position_rad", 0.0)
        ),
        gripper_open_position_rad=float(
            gripper.get("open_position_rad", -1.0471975511965976)
        ),
        left_gripper_closed_position_rad=calibrated_value(
            "left", "closed_position_rad"
        ),
        left_gripper_open_position_rad=calibrated_value(
            "left", "open_position_rad"
        ),
        right_gripper_closed_position_rad=calibrated_value(
            "right", "closed_position_rad"
        ),
        right_gripper_open_position_rad=calibrated_value(
            "right", "open_position_rad"
        ),
        kp=tuple(float(v) for v in gains.get("kp", DEFAULT_KP)),
        kd=tuple(float(v) for v in gains.get("kd", DEFAULT_KD)),
    )


def require_openarm_can() -> ModuleType:
    try:
        return importlib.import_module("openarm_can")
    except ImportError as exc:
        raise RuntimeError(
            "OpenArm real support is optional. Install the official C++ library "
            "and run `uv sync --extra openarm`."
        ) from exc


class OpenArmSide(Protocol):
    port: str

    def read_q(self) -> np.ndarray: ...

    def send(self, q: np.ndarray, gripper_opening: float) -> None: ...

    def close(self) -> None: ...


class OpenArmSdkSide:
    """One physical arm; all unstable SDK calls are contained here."""

    def __init__(
        self,
        port: str,
        *,
        enable_fd: bool,
        kp: tuple[float, ...],
        kd: tuple[float, ...],
        gripper_closed_position_rad: float = 0.0,
        gripper_open_position_rad: float = -1.0471975511965976,
        sdk: ModuleType | None = None,
    ) -> None:
        self.port = port
        self.sdk = sdk or require_openarm_can()
        self.kp = kp
        self.kd = kd
        self.gripper_closed_position_rad = float(gripper_closed_position_rad)
        self.gripper_open_position_rad = float(gripper_open_position_rad)
        motor_types = [
            self.sdk.MotorType.DM8009,
            self.sdk.MotorType.DM8009,
            self.sdk.MotorType.DM4340,
            self.sdk.MotorType.DM4340,
            self.sdk.MotorType.DM4310,
            self.sdk.MotorType.DM4310,
            self.sdk.MotorType.DM4310,
        ]
        self.arm = self.sdk.OpenArm(port, enable_fd)
        self.arm.init_arm_motors(motor_types, list(SEND_CAN_IDS), list(RECV_CAN_IDS))
        self.arm.init_gripper_motor(
            self.sdk.MotorType.DM4310,
            GRIPPER_SEND_CAN_ID,
            GRIPPER_RECV_CAN_ID,
            self.sdk.ControlMode.POS_FORCE,
        )
        self.arm.set_callback_mode_all(self.sdk.CallbackMode.STATE)
        self.arm.enable_all()
        self.arm.recv_all(2_000)

    def read_q(self) -> np.ndarray:
        self.arm.refresh_all()
        # CAN-FD responses can reach the USB adapter in separate batches.
        # Give all J1-J7 state frames time to queue before draining them;
        # otherwise untouched Motor objects still look like valid zeros.
        time.sleep(0.002)
        self.arm.recv_all(2_000)
        motors = self.arm.get_arm().get_motors()
        values = np.asarray(
            [motor.get_position() for motor in motors], dtype=np.float32
        )
        if values.shape != (ARM_DOF,):
            raise RuntimeError(
                f"OpenArm {self.port} returned {len(values)} joints; expected {ARM_DOF}."
            )
        if not np.all(np.isfinite(values)):
            raise RuntimeError(
                f"OpenArm {self.port} returned non-finite joint feedback."
            )
        return values

    def read_startup_q(self) -> np.ndarray:
        """Discard cold SDK samples and require a stable measured start pose."""
        samples: list[np.ndarray] = []
        for _ in range(6):
            samples.append(self.read_q())
            time.sleep(0.02)
        recent = np.stack(samples[-3:])
        excursions = np.ptp(recent, axis=0)
        joint = int(np.argmax(excursions))
        if float(excursions[joint]) > 0.1:
            raise RuntimeError(
                f"OpenArm {self.port} startup feedback is unstable at "
                f"joint{joint + 1} ({float(excursions[joint]):.3f} rad span)."
            )
        return np.median(recent, axis=0).astype(np.float32)

    def send(self, q: np.ndarray, gripper_opening: float) -> None:
        params = [
            self.sdk.MITParam(kp, kd, float(target), 0.0, 0.0)
            for kp, kd, target in zip(self.kp, self.kd, q, strict=True)
        ]
        self.arm.get_arm().mit_control_all(params)
        # HandUMI uses 0=closed and 1=open.  openarm_can >=1.2.6 expects the
        # raw J8 motor target; on OpenArm v1, closed is 0 rad and open is
        # -60 degrees on both arms (the official zero-calibration convention).
        opening = float(np.clip(gripper_opening, 0.0, 1.0))
        motor_position = self.gripper_closed_position_rad + opening * (
            self.gripper_open_position_rad - self.gripper_closed_position_rad
        )
        self.arm.get_gripper().set_position(motor_position)
        self.arm.recv_all(500)

    def close(self) -> None:
        self.arm.disable_all()
        self.arm.recv_all(1_000)


SideFactory = Callable[..., OpenArmSide]


class OpenArmJointStreamer:
    """Velocity-limited latest-target streamer with a stale-command hold."""

    def __init__(
        self,
        arms: dict[str, OpenArmSide],
        settings: OpenArmCanSettings,
        initial_q: dict[str, np.ndarray],
    ) -> None:
        self.arms = arms
        self.settings = settings
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="openarm-can", daemon=True
        )
        self._error: BaseException | None = None
        self._targets = {side: q.copy() for side, q in initial_q.items()}
        self._commanded = {side: q.copy() for side, q in initial_q.items()}
        self._feedback = {side: q.copy() for side, q in initial_q.items()}
        self._grippers = {side: 0.0 for side in SIDES}
        self._last_target_at = time.monotonic()
        self._max_speed = settings.max_joint_speed_rad_s
        self._waiting_for_targets = False

    def start(self) -> None:
        self._thread.start()

    def set_max_speed(self, value: float) -> None:
        with self._lock:
            self._max_speed = float(value)

    def set_targets(
        self,
        targets: dict[str, np.ndarray],
        grippers: dict[str, float] | None = None,
    ) -> None:
        self.raise_if_failed()
        with self._lock:
            for side, target in targets.items():
                q = np.asarray(target, dtype=np.float32)
                if q.shape != (ARM_DOF,) or not np.all(np.isfinite(q)):
                    raise ValueError(
                        f"Invalid OpenArm target for {side}: shape={q.shape}"
                    )
                self._targets[side] = q.copy()
            if grippers:
                self._grippers.update(
                    {
                        side: float(np.clip(value, 0.0, 1.0))
                        for side, value in grippers.items()
                    }
                )
            self._last_target_at = time.monotonic()

    def hold(self) -> dict[str, np.ndarray]:
        self.raise_if_failed()
        with self._lock:
            held = {side: q.copy() for side, q in self._feedback.items()}
            self._targets = {side: q.copy() for side, q in held.items()}
            self._commanded = {side: q.copy() for side, q in held.items()}
            self._last_target_at = time.monotonic()
        return held

    def feedback(self) -> dict[str, np.ndarray]:
        with self._lock:
            return {side: q.copy() for side, q in self._feedback.items()}

    def wait_until_targets(self, *, timeout_s: float, tolerance_rad: float) -> None:
        deadline = time.monotonic() + timeout_s
        # A blocking home move intentionally keeps one target for several
        # seconds. Do not let the stale-command watchdog cancel that target;
        # it is meant for an interrupted live teleop stream.
        with self._lock:
            expected = {side: self._targets[side].copy() for side in self.arms}
            self._waiting_for_targets = True
        try:
            while True:
                self.raise_if_failed()
                with self._lock:
                    joint_errors = {
                        side: np.abs(self._feedback[side] - expected[side])
                        for side in self.arms
                    }
                    worst_side = max(
                        joint_errors,
                        key=lambda side: float(np.max(joint_errors[side])),
                        default=None,
                    )
                    worst_joint = (
                        int(np.argmax(joint_errors[worst_side]))
                        if worst_side is not None
                        else 0
                    )
                    max_error = (
                        float(joint_errors[worst_side][worst_joint])
                        if worst_side is not None
                        else 0.0
                    )
                    measured = (
                        float(self._feedback[worst_side][worst_joint])
                        if worst_side is not None
                        else 0.0
                    )
                    target = (
                        float(expected[worst_side][worst_joint])
                        if worst_side is not None
                        else 0.0
                    )
                if max_error <= tolerance_rad:
                    return
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"OpenArm home timeout: {worst_side} joint{worst_joint + 1} "
                        f"error={max_error:.3f} rad, measured={measured:.3f} rad, "
                        f"target={target:.3f} rad."
                    )
                time.sleep(0.05)
        finally:
            with self._lock:
                self._waiting_for_targets = False
                self._last_target_at = time.monotonic()

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("OpenArm command streamer failed") from self._error

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.raise_if_failed()

    def _run(self) -> None:
        period = 1.0 / self.settings.command_rate_hz
        next_tick = time.monotonic()
        try:
            while not self._stop.is_set():
                now = time.monotonic()
                with self._lock:
                    if (
                        not self._waiting_for_targets
                        and now - self._last_target_at
                        > self.settings.watchdog_timeout_s
                    ):
                        self._targets = {
                            side: q.copy() for side, q in self._commanded.items()
                        }
                    step = self._max_speed * period
                    commands = {
                        side: self._commanded[side]
                        + np.clip(
                            self._targets[side] - self._commanded[side], -step, step
                        )
                        for side in self.arms
                    }
                    grippers = self._grippers.copy()
                    self._commanded = {side: q.copy() for side, q in commands.items()}

                feedback: dict[str, np.ndarray] = {}
                for side, arm in self.arms.items():
                    arm.send(commands[side], grippers[side])
                    feedback[side] = arm.read_q()
                    joint_errors = np.abs(feedback[side] - commands[side])
                    joint = int(np.argmax(joint_errors))
                    error = float(joint_errors[joint])
                    if error > self.settings.following_error_rad:
                        raise RuntimeError(
                            f"OpenArm {side} joint{joint + 1} following error "
                            f"{error:.3f} rad exceeds "
                            f"{self.settings.following_error_rad:.3f} rad."
                        )
                with self._lock:
                    self._feedback = feedback
                next_tick += period
                if (remaining := next_tick - time.monotonic()) > 0:
                    time.sleep(remaining)
                else:
                    next_tick = time.monotonic()
        except BaseException as exc:
            self._error = exc
            self._stop.set()
            log.error("OpenArm command streamer failed: %s", exc)


class OpenArmCanEnvironment:
    """Bimanual OpenArm backend implementing the generic teleop contract."""

    def __init__(
        self,
        settings: OpenArmCanSettings,
        *,
        side_factory: SideFactory = OpenArmSdkSide,
        active_sides: tuple[str, ...] = SIDES,
        joint_limits: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        if not active_sides or any(side not in SIDES for side in active_sides):
            raise ValueError(f"Invalid OpenArm active sides: {active_sides}")
        self.settings = settings
        self.side_factory = side_factory
        self.active_sides = tuple(dict.fromkeys(active_sides))
        self.joint_limits = joint_limits or {}
        self.arms: dict[str, OpenArmSide] = {}
        self.streamer: OpenArmJointStreamer | None = None
        self._last_limit_warning_at = {side: 0.0 for side in SIDES}

    def connect(self) -> None:
        if self.arms:
            return
        ports = {"left": self.settings.left_port, "right": self.settings.right_port}
        for side in self.active_sides:
            port = ports[side]
            closed = getattr(
                self.settings, f"{side}_gripper_closed_position_rad"
            )
            open_ = getattr(self.settings, f"{side}_gripper_open_position_rad")
            log.info("Connecting OpenArm %s on %s.", side, port)
            self.arms[side] = self.side_factory(
                port,
                enable_fd=self.settings.enable_fd,
                kp=self.settings.kp,
                kd=self.settings.kd,
                gripper_closed_position_rad=(
                    self.settings.gripper_closed_position_rad
                    if closed is None
                    else closed
                ),
                gripper_open_position_rad=(
                    self.settings.gripper_open_position_rad
                    if open_ is None
                    else open_
                ),
            )

    def prepare(self, *, repair: bool = True) -> None:
        from handumi.real.can_setup import ensure_can_fd_interfaces_ready

        ensure_can_fd_interfaces_ready(
            [
                self.settings.left_port if side == "left" else self.settings.right_port
                for side in self.active_sides
            ],
            bitrate=self.settings.bitrate,
            dbitrate=self.settings.dbitrate,
            repair=repair,
        )

    def _split_side_q(
        self, q: np.ndarray, joint_names: list[str], side: str
    ) -> np.ndarray:
        names = [f"openarm_{side}_joint{i}" for i in range(1, ARM_DOF + 1)]
        values = np.asarray(
            [q[joint_names.index(name)] for name in names], dtype=np.float32
        )
        for index, (name, value) in enumerate(zip(names, values, strict=True)):
            limits = self.joint_limits.get(name)
            if limits is None:
                continue
            lower, upper = limits
            scalar = float(value)
            if (
                scalar < lower - JOINT_LIMIT_SNAP_TOLERANCE_RAD
                or scalar > upper + JOINT_LIMIT_SNAP_TOLERANCE_RAD
            ):
                raise ValueError(
                    f"OpenArm target {name}={scalar:.8f} is outside "
                    f"URDF limits [{lower:.8f}, {upper:.8f}]."
                )
            # IK and float32 conversion can differ from a decimal URDF
            # limit by a few microradians. Snap only that numerical fringe.
            values[index] = np.clip(scalar, lower, upper)
        return values

    def _split_q(self, q: np.ndarray, joint_names: list[str]) -> dict[str, np.ndarray]:
        return {
            side: self._split_side_q(q, joint_names, side)
            for side in self.active_sides
        }

    @staticmethod
    def _merge_q(
        arm_q: dict[str, np.ndarray], base_q: np.ndarray, joint_names: list[str]
    ) -> np.ndarray:
        q = np.asarray(base_q, dtype=np.float32).copy()
        for side, values in arm_q.items():
            for i, value in enumerate(values, start=1):
                q[joint_names.index(f"openarm_{side}_joint{i}")] = value
        return q

    def home(self, q: np.ndarray, joint_names: list[str]) -> None:
        if not self.arms:
            raise RuntimeError("connect() before home()")
        initial = {
            side: getattr(arm, "read_startup_q", arm.read_q)()
            for side, arm in self.arms.items()
        }
        for side, measured in initial.items():
            log.info(
                "OpenArm %s measured startup joints (deg): %s",
                side,
                np.round(np.rad2deg(measured), 1).tolist(),
            )
        self.streamer = OpenArmJointStreamer(self.arms, self.settings, initial)
        self.streamer.start()
        self.move_home(q, joint_names)

    def move_home(self, q: np.ndarray, joint_names: list[str]) -> None:
        if self.streamer is None:
            raise RuntimeError("home() before move_home()")
        targets = self._split_q(q, joint_names)
        for side, target in targets.items():
            log.info(
                "Moving OpenArm %s slowly to home (deg): %s at %.1f deg/s max.",
                side,
                np.round(np.rad2deg(target), 1).tolist(),
                float(np.rad2deg(self.settings.home_max_joint_speed_rad_s)),
            )
        self.streamer.set_max_speed(self.settings.home_max_joint_speed_rad_s)
        try:
            # For the collision-safe forward pose, establish shoulder/elbow
            # lateral clearance while retaining the measured distal posture.
            # Only after both arms are spread do we bend J4 toward 90 degrees.
            # ``down`` and ``arms_90`` remain exact diagnostic poses and skip
            # this waypoint because their first three target joints are zero.
            if any(np.any(np.abs(target[:3]) > 1e-4) for target in targets.values()):
                feedback = self.streamer.feedback()
                clearance_targets = {
                    side: np.concatenate((target[:3], feedback[side][3:])).astype(
                        np.float32
                    )
                    for side, target in targets.items()
                }
                log.info(
                    "Opening OpenArm shoulders first to clear the center structure."
                )
                self.streamer.set_targets(
                    clearance_targets,
                    {side: 0.0 for side in self.active_sides},
                )
                self.streamer.wait_until_targets(
                    timeout_s=self.settings.home_timeout_s,
                    tolerance_rad=self.settings.home_tolerance_rad,
                )
            self.streamer.set_targets(
                targets,
                # Start from a deterministic, safe closed gripper. Live
                # Feetech openings take over after startup.
                {side: 0.0 for side in self.active_sides},
            )
            self.streamer.wait_until_targets(
                timeout_s=self.settings.home_timeout_s,
                tolerance_rad=self.settings.home_tolerance_rad,
            )
        finally:
            self.streamer.set_max_speed(self.settings.max_joint_speed_rad_s)

    def command(
        self,
        q: np.ndarray,
        joint_names: list[str],
        gripper_openings: dict[str, float],
    ) -> None:
        if self.streamer is None:
            raise RuntimeError("home() before command()")
        targets: dict[str, np.ndarray] = {}
        for side in self.active_sides:
            try:
                targets[side] = self._split_side_q(q, joint_names, side)
            except ValueError as exc:
                # Do not clamp individual joints: that changes the solved arm
                # posture and can create surprising Cartesian motion. Drop the
                # whole unsafe arm target instead; the streamer keeps its last
                # safe target and automatically resumes once IK returns inside
                # the URDF limits. Rate-limit warnings during a sustained hold.
                now = time.monotonic()
                if now - self._last_limit_warning_at[side] >= 1.0:
                    log.warning(
                        "%s Holding the previous safe %s-arm target.", exc, side
                    )
                    self._last_limit_warning_at[side] = now
        self.streamer.set_targets(targets, gripper_openings)

    def hold(self, base_q: np.ndarray, joint_names: list[str]) -> np.ndarray:
        if self.streamer is None:
            raise RuntimeError("home() before hold()")
        return self._merge_q(self.streamer.hold(), base_q, joint_names)

    def check_health(self) -> None:
        if self.streamer is not None:
            self.streamer.raise_if_failed()

    def close(self) -> None:
        error: BaseException | None = None
        if self.streamer is not None:
            try:
                self.streamer.stop()
            except BaseException as exc:  # still disable every arm
                error = exc
        for side, arm in list(self.arms.items()):
            try:
                arm.close()
            except Exception as exc:  # pragma: no cover - hardware cleanup
                log.warning("Failed to disable OpenArm %s: %s", side, exc)
        self.arms.clear()
        self.streamer = None
        if error is not None:
            raise error


__all__ = [
    "ARM_DOF",
    "OpenArmCanEnvironment",
    "OpenArmCanSettings",
    "OpenArmJointStreamer",
    "load_openarm_settings",
    "require_openarm_can",
]
