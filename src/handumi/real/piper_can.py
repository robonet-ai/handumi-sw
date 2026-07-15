"""AgileX Piper CAN backend used by real HandUMI teleop.

The teleop script computes one IK configuration ``q`` at the live tracking
rate. This module turns that ``q`` into Piper SDK joint units and streams the
latest target on a fixed-rate CAN thread so the robot receives smooth,
bounded joint commands even if the IK loop jitters.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
import yaml

from handumi.config import DEFAULT_RIG_CONFIG, EXAMPLE_RIG_CONFIG
from handumi.robots.registry import RobotRealConfig

log = logging.getLogger("handumi.real.piper")

RAD_TO_MDEG = 1000.0 * 180.0 / np.pi
MDEG_TO_RAD = 1.0 / RAD_TO_MDEG
ARM_JOINT_COUNT = 6
SIDE_NAMES = ("left", "right")


class PiperArm(Protocol):
    """Small interface implemented by the SDK wrapper and fake test arms."""

    port: str

    def read_mdeg(self) -> np.ndarray: ...
    def send_mdeg(self, cmd: np.ndarray) -> None: ...
    def send_gripper_microm(self, opening_microm: int, effort: int) -> None: ...
    def disconnect(self) -> None: ...


ArmFactory = Callable[[str, int, float, int], PiperArm]


@dataclass(frozen=True)
class PiperCanSettings:
    """Resolved Piper real-teleop settings from robot defaults + local rig."""

    left_port: str
    right_port: str
    bitrate: int = 1_000_000
    restart_ms: int = 100
    command_rate_hz: float = 100.0
    max_joint_speed_deg_s: float = 180.0
    home_max_joint_speed_deg_s: float = 20.0
    home_timeout_s: float = 30.0
    home_tolerance_deg: float = 3.0
    speed_percent: int = 80
    enable_timeout_s: float = 10.0
    gripper_effort: int = 1000


def load_piper_can_settings(
    rig_config: Path = DEFAULT_RIG_CONFIG,
    real_config: RobotRealConfig | None = None,
) -> PiperCanSettings:
    """Load Piper CAN ports from ``rig.yaml`` and command defaults from robot YAML."""
    if not rig_config.exists():
        raise SystemExit(
            f"Missing rig configuration: {rig_config}.\n"
            f"Create it with: cp {EXAMPLE_RIG_CONFIG} {DEFAULT_RIG_CONFIG}"
        )
    with rig_config.open("r", encoding="utf-8") as handle:
        rig: dict[str, Any] = yaml.safe_load(handle) or {}

    can = (((rig.get("robots") or {}).get("piper") or {}).get("can") or {})
    if not isinstance(can, dict):
        raise SystemExit(f"Missing or invalid 'robots.piper.can' section in {rig_config}.")
    missing = [key for key in ("left_port", "right_port") if not can.get(key)]
    if missing:
        raise SystemExit(
            f"Missing Piper CAN setting(s) in {rig_config}: {', '.join(missing)}."
        )

    defaults = real_config or RobotRealConfig()
    return PiperCanSettings(
        left_port=str(can["left_port"]),
        right_port=str(can["right_port"]),
        bitrate=int(can.get("bitrate", 1_000_000)),
        restart_ms=int(can.get("restart_ms", 100)),
        command_rate_hz=defaults.command_rate_hz,
        max_joint_speed_deg_s=defaults.max_joint_speed_deg_s,
        home_max_joint_speed_deg_s=defaults.home_max_joint_speed_deg_s,
        home_timeout_s=defaults.home_timeout_s,
        home_tolerance_deg=defaults.home_tolerance_deg,
        speed_percent=defaults.speed_percent,
        gripper_effort=defaults.gripper_effort,
    )


def piper_arm_joint_indices(actuated_names: list[str] | tuple[str, ...], side: str) -> list[int]:
    """Return the six Piper arm-joint indices for ``side`` in URDF order."""
    if side not in SIDE_NAMES:
        raise ValueError(f"expected side in {SIDE_NAMES}, got {side!r}")
    names = list(actuated_names)
    wanted = [f"{side}_joint{i}" for i in range(1, ARM_JOINT_COUNT + 1)]
    missing = [name for name in wanted if name not in names]
    if missing:
        raise ValueError(f"missing Piper joints in URDF: {', '.join(missing)}")
    return [names.index(name) for name in wanted]


def q_to_piper_mdeg(
    q: np.ndarray,
    actuated_names: list[str] | tuple[str, ...],
) -> dict[str, np.ndarray]:
    """Convert full robot ``q`` in radians to Piper SDK milli-degree joints."""
    q_arr = np.asarray(q, dtype=np.float64).reshape(-1)
    return {
        side: np.rint(q_arr[piper_arm_joint_indices(actuated_names, side)] * RAD_TO_MDEG)
        .astype(np.int64)
        .reshape(ARM_JOINT_COUNT)
        for side in SIDE_NAMES
    }


def piper_mdeg_to_q(
    *,
    left_mdeg: np.ndarray,
    right_mdeg: np.ndarray,
    actuated_names: list[str] | tuple[str, ...],
    base_q: np.ndarray,
) -> np.ndarray:
    """Write real Piper feedback milli-degrees into a full robot ``q`` vector."""
    q = np.asarray(base_q, dtype=np.float32).copy()
    for side, values in (("left", left_mdeg), ("right", right_mdeg)):
        indices = piper_arm_joint_indices(actuated_names, side)
        q[indices] = np.asarray(values, dtype=np.float32)[:ARM_JOINT_COUNT] * MDEG_TO_RAD
    return q


def step_mdeg_toward(
    current: np.ndarray,
    target: np.ndarray,
    max_step_mdeg: float,
) -> np.ndarray:
    """Move one command sample toward ``target`` by at most ``max_step_mdeg`` per joint."""
    current_f = np.asarray(current, dtype=np.float64)
    target_f = np.asarray(target, dtype=np.float64)
    if max_step_mdeg <= 0.0:
        return np.rint(target_f).astype(np.int64)
    delta = np.clip(target_f - current_f, -float(max_step_mdeg), float(max_step_mdeg))
    return np.rint(current_f + delta).astype(np.int64)


def format_mdeg(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{int(v):7d}" for v in np.asarray(values).reshape(-1)) + "]"


class PiperSdkArm:
    """Thin piper_sdk wrapper for one physical Piper arm."""

    def __init__(
        self,
        port: str,
        speed_percent: int,
        enable_timeout_s: float,
        gripper_effort: int,
    ) -> None:
        try:
            from piper_sdk import C_PiperInterface_V2
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Missing piper_sdk. Install real Piper support with: uv sync --extra piper"
            ) from exc

        self.port = port
        self.speed_percent = int(speed_percent)
        self.gripper_effort = int(gripper_effort)
        self.arm = C_PiperInterface_V2(port)
        self.arm.ConnectPort()
        time.sleep(0.2)

        status = self.arm.GetArmStatus().arm_status
        if status.motion_status != 0:
            self.arm.EmergencyStop(0x02)
            time.sleep(0.1)
        if status.ctrl_mode == 2:
            log.info("[%s] Arm is in teaching mode; sending resume.", self.port)
            self.arm.EmergencyStop(0x02)
            time.sleep(0.1)

        deadline = time.time() + float(enable_timeout_s)
        while not self.arm.EnablePiper():
            if time.time() > deadline:
                raise TimeoutError(f"{self.port}: timed out enabling Piper")
            time.sleep(0.02)
        self.set_joint_mode()

    def set_joint_mode(self) -> None:
        self.arm.MotionCtrl_2(0x01, 0x01, self.speed_percent, 0x00)

    def read_mdeg(self) -> np.ndarray:
        joint_state = self.arm.GetArmJointMsgs().joint_state
        return np.array(
            [
                joint_state.joint_1,
                joint_state.joint_2,
                joint_state.joint_3,
                joint_state.joint_4,
                joint_state.joint_5,
                joint_state.joint_6,
            ],
            dtype=np.int64,
        )

    def send_mdeg(self, cmd: np.ndarray) -> None:
        values = [int(v) for v in np.asarray(cmd, dtype=np.int64)[:ARM_JOINT_COUNT]]
        self.arm.JointCtrl(*values)

    def send_gripper_microm(self, opening_microm: int, effort: int) -> None:
        self.arm.GripperCtrl(int(opening_microm), int(effort), 0x01, 0)

    def disconnect(self) -> None:
        disconnect = getattr(self.arm, "DisconnectPort", None)
        if disconnect is not None:
            disconnect()


class PiperJointStreamer:
    """Latest-target, fixed-rate sender for both real Piper arms."""

    def __init__(
        self,
        arms: dict[str, PiperArm],
        *,
        command_rate_hz: float,
        max_joint_speed_deg_s: float,
        gripper_effort: int,
    ) -> None:
        if command_rate_hz <= 0.0:
            raise ValueError("command_rate_hz must be > 0")
        if max_joint_speed_deg_s <= 0.0:
            raise ValueError("max_joint_speed_deg_s must be > 0")
        if not arms:
            raise ValueError("no Piper arms connected")

        self.arms = arms
        self.command_rate_hz = float(command_rate_hz)
        self.gripper_effort = int(gripper_effort)
        self.max_step_mdeg = max(1.0, max_joint_speed_deg_s * 1000.0 / command_rate_hz)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._commanded = {
            side: arm.read_mdeg().astype(np.int64) for side, arm in self.arms.items()
        }
        self._targets = {side: cmd.copy() for side, cmd in self._commanded.items()}
        self._gripper_targets: dict[str, int | None] = {side: None for side in self.arms}
        self._thread = threading.Thread(
            target=self._run,
            name="handumi-piper-can-streamer",
            daemon=True,
        )

    def start(self) -> None:
        log.info(
            "Piper stream: %.1f Hz, %.3f deg/tick",
            self.command_rate_hz,
            self.max_step_mdeg / 1000.0,
        )
        self._thread.start()

    def set_max_joint_speed_deg_s(self, max_joint_speed_deg_s: float) -> None:
        if max_joint_speed_deg_s <= 0.0:
            raise ValueError("max_joint_speed_deg_s must be > 0")
        with self._lock:
            self.max_step_mdeg = max(
                1.0,
                max_joint_speed_deg_s * 1000.0 / self.command_rate_hz,
            )
        log.info("Piper stream max step: %.3f deg/tick", self.max_step_mdeg / 1000.0)

    def set_targets(self, targets: dict[str, np.ndarray]) -> None:
        self.raise_if_failed()
        with self._lock:
            for side, target in targets.items():
                if side in self._targets:
                    self._targets[side] = (
                        np.asarray(target, dtype=np.int64)[:ARM_JOINT_COUNT].copy()
                    )

    def set_gripper_targets_microm(self, targets: dict[str, int | None]) -> None:
        self.raise_if_failed()
        with self._lock:
            for side, target in targets.items():
                if side in self._gripper_targets:
                    self._gripper_targets[side] = None if target is None else max(0, int(target))

    def latest_commands(self) -> dict[str, np.ndarray]:
        with self._lock:
            return {side: cmd.copy() for side, cmd in self._commanded.items()}

    def hold_current_commands(self) -> dict[str, np.ndarray]:
        """Cancel pending motion and hold the latest scheduled joint commands."""
        self.raise_if_failed()
        with self._lock:
            held = {side: cmd.copy() for side, cmd in self._commanded.items()}
            self._targets = {side: cmd.copy() for side, cmd in held.items()}
        return held

    def feedback_mdeg(self) -> dict[str, np.ndarray]:
        return {side: arm.read_mdeg().astype(np.int64) for side, arm in self.arms.items()}

    def max_command_error_mdeg(self) -> float:
        with self._lock:
            errors = [
                float(np.max(np.abs(self._commanded[side] - target)))
                for side, target in self._targets.items()
            ]
        return max(errors, default=0.0)

    def max_feedback_error_mdeg(self) -> float:
        with self._lock:
            targets = {side: target.copy() for side, target in self._targets.items()}
        errors = []
        for side, target in targets.items():
            feedback = self.arms[side].read_mdeg()
            errors.append(float(np.max(np.abs(feedback - target))))
        return max(errors, default=0.0)

    def wait_until_targets(
        self,
        *,
        timeout_s: float,
        tolerance_mdeg: float,
        print_period_s: float = 0.5,
    ) -> None:
        deadline = time.perf_counter() + float(timeout_s)
        last_print = 0.0
        while True:
            self.raise_if_failed()
            cmd_error = self.max_command_error_mdeg()
            feedback_error = self.max_feedback_error_mdeg()
            if max(cmd_error, feedback_error) <= tolerance_mdeg:
                return
            now = time.perf_counter()
            if now > deadline:
                raise TimeoutError(
                    "timed out waiting for Piper target "
                    f"(cmd_err={cmd_error / 1000.0:.2f}deg, "
                    f"feedback_err={feedback_error / 1000.0:.2f}deg)"
                )
            if now - last_print >= print_period_s:
                log.info(
                    "Piper homing: cmd_err=%.2fdeg feedback_err=%.2fdeg",
                    cmd_error / 1000.0,
                    feedback_error / 1000.0,
                )
                last_print = now
            time.sleep(0.05)

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("Piper command streamer failed") from self._error

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.raise_if_failed()

    def _run(self) -> None:
        period = 1.0 / self.command_rate_hz
        next_time = time.perf_counter()
        try:
            while not self._stop.is_set():
                with self._lock:
                    gripper_targets = self._gripper_targets.copy()
                    next_commands = {
                        side: step_mdeg_toward(
                            self._commanded[side],
                            self._targets[side],
                            self.max_step_mdeg,
                        )
                        for side in self.arms
                    }
                    # Publish the scheduled command before sending. A concurrent
                    # hold then captures this exact tick and prevents later motion.
                    self._commanded.update(
                        {side: cmd.copy() for side, cmd in next_commands.items()}
                    )

                for side, arm in self.arms.items():
                    cmd = next_commands[side]
                    arm.send_mdeg(cmd)
                    gripper = gripper_targets.get(side)
                    if gripper is not None:
                        arm.send_gripper_microm(gripper, self.gripper_effort)

                next_time += period
                remaining = next_time - time.perf_counter()
                if remaining > 0.0:
                    time.sleep(remaining)
        except BaseException as exc:
            self._error = exc
            self._stop.set()
            log.error("Piper command streamer failed: %s", exc)


class PiperCanEnvironment:
    """Owns Piper CAN arms plus the smooth latest-target streamer."""

    def __init__(
        self,
        settings: PiperCanSettings,
        *,
        arm_factory: ArmFactory = PiperSdkArm,
    ) -> None:
        self.settings = settings
        self.arm_factory = arm_factory
        self.arms: dict[str, PiperArm] = {}
        self.streamer: PiperJointStreamer | None = None

    def connect(self) -> None:
        if self.arms:
            return
        for side, port in (("left", self.settings.left_port), ("right", self.settings.right_port)):
            log.info("Connecting Piper %s on %s.", side, port)
            self.arms[side] = self.arm_factory(
                port,
                self.settings.speed_percent,
                self.settings.enable_timeout_s,
                self.settings.gripper_effort,
            )

    def home(self, home_targets_mdeg: dict[str, np.ndarray]) -> None:
        if not self.arms:
            raise RuntimeError("connect() before home()")
        self.streamer = PiperJointStreamer(
            self.arms,
            command_rate_hz=self.settings.command_rate_hz,
            max_joint_speed_deg_s=self.settings.home_max_joint_speed_deg_s,
            gripper_effort=self.settings.gripper_effort,
        )
        self.streamer.start()
        self.move_home(home_targets_mdeg)

    def move_home(self, home_targets_mdeg: dict[str, np.ndarray]) -> None:
        """Move to home at the configured slow limit, then restore teleop speed."""
        if self.streamer is None:
            raise RuntimeError("home() before move_home()")
        log.info(
            "Homing Piper to XHUMAN pose: left=%s right=%s",
            format_mdeg(home_targets_mdeg["left"]),
            format_mdeg(home_targets_mdeg["right"]),
        )
        self.streamer.set_max_joint_speed_deg_s(
            self.settings.home_max_joint_speed_deg_s
        )
        try:
            self.streamer.set_targets(home_targets_mdeg)
            self.streamer.wait_until_targets(
                timeout_s=self.settings.home_timeout_s,
                tolerance_mdeg=self.settings.home_tolerance_deg * 1000.0,
            )
            log.info("Piper home reached.")
        finally:
            self.streamer.set_max_joint_speed_deg_s(
                self.settings.max_joint_speed_deg_s
            )

    def set_q(self, q: np.ndarray, actuated_names: list[str] | tuple[str, ...]) -> None:
        self.set_targets(q_to_piper_mdeg(q, actuated_names))

    def set_gripper_widths_mm(self, widths_mm: dict[str, float]) -> None:
        if self.streamer is None:
            raise RuntimeError("home() before set_gripper_widths_mm()")
        targets = {
            side: int(round(max(0.0, float(width_mm)) * 1000.0))
            for side, width_mm in widths_mm.items()
        }
        self.streamer.set_gripper_targets_microm(targets)

    def set_targets(self, targets_mdeg: dict[str, np.ndarray]) -> None:
        if self.streamer is None:
            raise RuntimeError("home() before set_targets()")
        self.streamer.set_targets(targets_mdeg)

    def latest_commands_mdeg(self) -> dict[str, np.ndarray]:
        if self.streamer is None:
            return {}
        return self.streamer.latest_commands()

    def hold_current_commands_mdeg(self) -> dict[str, np.ndarray]:
        if self.streamer is None:
            raise RuntimeError("home() before hold_current_commands_mdeg()")
        return self.streamer.hold_current_commands()

    def feedback_mdeg(self) -> dict[str, np.ndarray]:
        return {side: arm.read_mdeg().astype(np.int64) for side, arm in self.arms.items()}

    def raise_if_failed(self) -> None:
        if self.streamer is not None:
            self.streamer.raise_if_failed()

    def close(self) -> None:
        try:
            if self.streamer is not None:
                self.streamer.stop()
        finally:
            for arm in self.arms.values():
                try:
                    arm.disconnect()
                except Exception as exc:  # pragma: no cover - defensive hardware cleanup
                    log.warning("Failed to disconnect Piper %s: %s", arm.port, exc)
            self.arms.clear()
            self.streamer = None


__all__ = [
    "MDEG_TO_RAD",
    "RAD_TO_MDEG",
    "PiperCanEnvironment",
    "PiperCanSettings",
    "PiperJointStreamer",
    "PiperSdkArm",
    "format_mdeg",
    "load_piper_can_settings",
    "piper_arm_joint_indices",
    "piper_mdeg_to_q",
    "q_to_piper_mdeg",
    "step_mdeg_toward",
]
