import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from handumi.real.openarm_can import load_openarm_settings
from handumi.real.openarm_gripper_calibration import (
    OpenArmGripperLimits,
    load_openarm_gripper_limits,
    save_openarm_gripper_limits,
)
from handumi.scripts.setup.calibrate_openarm_grippers import (
    _read_stable_position,
    calibrate_side,
    calibrate_side_manually,
    parse_args,
)


class _Motor:
    def __init__(self):
        self.position = -0.5
        self.velocity = 0.0
        self.torque = 0.0

    def get_position(self):
        return self.position

    def get_velocity(self):
        return self.velocity

    def get_torque(self):
        return self.torque


class _MotorSnapshot:
    """Match openarm_can: get_motors() returns Motor values by copy."""

    def __init__(self, motor):
        self.position = motor.position
        self.velocity = motor.velocity
        self.torque = motor.torque

    def get_position(self):
        return self.position

    def get_velocity(self):
        return self.velocity

    def get_torque(self):
        return self.torque


class _Gripper:
    def __init__(self, motor):
        self.motor = motor

    def get_motors(self):
        return [_MotorSnapshot(self.motor)]

    def mit_control_one(self, _index, param):
        requested = float(param[2])
        position = float(np.clip(requested, -1.1, 0.0))
        stopped = not np.isclose(requested, position)
        self.motor.position = position
        self.motor.velocity = 0.0 if stopped else 1.0
        self.motor.torque = 0.5 if stopped else 0.0


class _Arm:
    last = None

    def __init__(self, port, enable_fd):
        self.port = port
        self.enable_fd = enable_fd
        self.motor = _Motor()
        self.gripper = _Gripper(self.motor)
        self.init_args = None
        self.disabled = False
        _Arm.last = self

    def init_gripper_motor(self, *args):
        self.init_args = args

    def set_callback_mode_all(self, _mode):
        pass

    def enable_all(self):
        pass

    def disable_all(self):
        self.disabled = True

    def recv_all(self, _timeout):
        pass

    def refresh_all(self):
        pass

    def get_gripper(self):
        return self.gripper


class _Sdk:
    class MotorType:
        DM4310 = "DM4310"

    class ControlMode:
        MIT = "MIT"

    class CallbackMode:
        STATE = "STATE"

    OpenArm = _Arm
    MITParam = staticmethod(lambda *args: args)


def test_calibrator_measures_both_stops_using_only_j8():
    with mock.patch(
        "handumi.scripts.setup.calibrate_openarm_grippers.time.sleep"
    ):
        limits = calibrate_side(
            port="can0",
            sdk=_Sdk,
            step_rad=0.1,
            max_travel_rad=1.5,
        )

    assert np.isclose(limits.closed_position_rad, 0.0)
    assert np.isclose(limits.open_position_rad, -1.1)
    assert _Arm.last.init_args == ("DM4310", 0x08, 0x18, "MIT")
    assert _Arm.last.disabled


def test_default_manual_calibration_reads_user_placed_endpoints_without_enabling():
    positions = iter((-0.01, -1.06))

    def place_gripper(_prompt):
        _Arm.last.motor.position = next(positions)
        return ""

    with mock.patch(
        "handumi.scripts.setup.calibrate_openarm_grippers.time.sleep"
    ):
        limits = calibrate_side_manually(
            port="can0", sdk=_Sdk, side="right", input_fn=place_gripper
        )

    assert np.isclose(limits.closed_position_rad, -0.01)
    assert np.isclose(limits.open_position_rad, -1.06)
    assert _Arm.last.disabled


def test_cli_defaults_to_manual_endpoint_capture():
    args = parse_args(["--side", "right"])
    assert not args.automatic


def test_manual_capture_discards_feedback_from_previous_endpoint():
    values = iter([0.97, 0.97, 0.97, -0.20, -0.20] + [-0.20] * 20)

    class Snapshot:
        def __init__(self, value):
            self.value = value

        def get_position(self):
            return self.value

    class Gripper:
        def get_motors(self):
            return [Snapshot(next(values))]

    class Arm:
        def refresh_all(self):
            pass

        def recv_all(self, _timeout):
            pass

        def get_gripper(self):
            return Gripper()

    with mock.patch(
        "handumi.scripts.setup.calibrate_openarm_grippers.time.sleep"
    ):
        captured = _read_stable_position(Arm())

    assert np.isclose(captured, -0.20)


def test_calibration_round_trip_preserves_the_other_side():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "openarm.yaml"
        save_openarm_gripper_limits("left", OpenArmGripperLimits(0.01, -1.02), path)
        save_openarm_gripper_limits("right", OpenArmGripperLimits(-0.02, -1.09), path)
        loaded = load_openarm_gripper_limits(path)

    assert set(loaded) == {"left", "right"}
    assert loaded["left"] == OpenArmGripperLimits(0.01, -1.02)
    assert loaded["right"] == OpenArmGripperLimits(-0.02, -1.09)


def test_runtime_loads_per_side_gripper_endpoints():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        calibration = root / "openarm.yaml"
        save_openarm_gripper_limits(
            "right", OpenArmGripperLimits(0.03, -1.00), calibration
        )
        settings = load_openarm_settings(
            root / "missing-rig.yaml", {}, calibration
        )

    assert settings.right_gripper_closed_position_rad == 0.03
    assert settings.right_gripper_open_position_rad == -1.0
    assert settings.left_gripper_open_position_rad is None


def test_invalid_v1_direction_is_rejected():
    with pytest.raises(ValueError, match="below"):
        OpenArmGripperLimits(0.0, 1.0).validate()
