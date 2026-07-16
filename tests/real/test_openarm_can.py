import tempfile
import time
from pathlib import Path
from unittest import mock

import numpy as np

from handumi.real.openarm_can import (
    OpenArmCanEnvironment,
    OpenArmCanSettings,
    OpenArmSdkSide,
    load_openarm_settings,
)
from handumi.robots.registry import load_embodiment


class FakeSide:
    instances: list["FakeSide"] = []
    port: str

    def __init__(self, port, **_kwargs):
        self.port = port
        self.q = np.zeros(7, dtype=np.float32)
        self.gripper = 1.0
        self.closed = False
        self.sent = []
        self.instances.append(self)

    def read_q(self):
        return self.q.copy()

    def send(self, q, gripper_opening):
        self.q = np.asarray(q, dtype=np.float32).copy()
        self.gripper = float(gripper_opening)
        self.sent.append(self.q.copy())

    def close(self):
        self.closed = True


class FakeSdkArm:
    def __init__(self, _port, _enable_fd):
        self.gripper_init = None
        self.gripper = mock.Mock()

    def init_arm_motors(self, *_args):
        pass

    def init_gripper_motor(self, *args):
        self.gripper_init = args

    def set_callback_mode_all(self, _mode):
        pass

    def enable_all(self):
        pass

    def recv_all(self, _timeout):
        pass

    def get_arm(self):
        return mock.Mock()

    def get_gripper(self):
        return self.gripper


class FakeSdk:
    class MotorType:
        DM8009 = "DM8009"
        DM4340 = "DM4340"
        DM4310 = "DM4310"

    class ControlMode:
        POS_FORCE = "POS_FORCE"

    class CallbackMode:
        STATE = "STATE"

    MITParam = staticmethod(lambda *args: args)
    OpenArm = FakeSdkArm


def test_sdk_side_initializes_gripper_for_posforce_position_commands():
    side = OpenArmSdkSide(
        "can0",
        enable_fd=True,
        kp=(1.0,) * 7,
        kd=(1.0,) * 7,
        sdk=FakeSdk,
    )

    assert side.arm.gripper_init == ("DM4310", 0x08, 0x18, "POS_FORCE")


def test_sdk_side_maps_normalized_opening_to_v1_negative_sixty_degree_range():
    side = OpenArmSdkSide(
        "can0",
        enable_fd=True,
        kp=(1.0,) * 7,
        kd=(1.0,) * 7,
        gripper_closed_position_rad=0.0,
        gripper_open_position_rad=-np.pi / 3.0,
        sdk=FakeSdk,
    )
    side.arm.get_arm().mit_control_all = mock.Mock()

    side.send(np.zeros(7, dtype=np.float32), 1.0)

    commanded = side.arm.gripper.set_position.call_args.args[0]
    assert np.isclose(commanded, -np.pi / 3.0)


def test_sdk_side_discards_cold_feedback_before_startup_pose():
    side = object.__new__(OpenArmSdkSide)
    side.port = "can0"
    actual = np.array([0.1, -0.2, 0.3, 1.2, 0.0, 0.1, -0.4], dtype=np.float32)
    side.read_q = mock.Mock(  # type: ignore[method-assign]
        side_effect=[np.zeros(7), np.zeros(7), actual, actual, actual, actual]
    )

    with mock.patch("handumi.real.openarm_can.time.sleep"):
        measured = side.read_startup_q()

    np.testing.assert_allclose(measured, actual)
    assert side.read_q.call_count == 6


def test_settings_combine_rig_ports_and_robot_control_defaults():
    with tempfile.TemporaryDirectory() as tmp:
        rig = Path(tmp) / "rig.yaml"
        rig.write_text(
            "robots:\n"
            "  openarmv1:\n"
            "    can:\n"
            "      fd: true\n"
            "      bitrate: 1000000\n"
            "      dbitrate: 5000000\n"
            "      left_port: can7\n"
            "      right_port: can6\n",
            encoding="utf-8",
        )
        settings = load_openarm_settings(
            rig,
            {"control": {"command_rate_hz": 80, "watchdog_timeout_s": 0.2}},
        )

    assert settings.left_port == "can7"
    assert settings.right_port == "can6"
    assert settings.enable_fd
    assert settings.command_rate_hz == 80
    assert settings.watchdog_timeout_s == 0.2
    assert settings.home_max_joint_speed_rad_s == 0.25
    assert np.isclose(settings.gripper_open_position_rad, -np.pi / 3.0)


def test_environment_streams_holds_and_disables_both_arms():
    FakeSide.instances.clear()
    settings = OpenArmCanSettings(
        command_rate_hz=500.0,
        max_joint_speed_rad_s=100.0,
        home_max_joint_speed_rad_s=100.0,
        watchdog_timeout_s=0.1,
    )
    environment = OpenArmCanEnvironment(settings, side_factory=FakeSide)
    runtime = load_embodiment("openarmv1")
    names = list(runtime.joint_names)
    home = runtime.home_q("down")

    environment.connect()
    environment.home(home, names)
    target = home.copy()
    target[names.index("openarm_left_joint1")] = 0.2
    target[names.index("openarm_right_joint2")] = -0.1
    environment.command(target, names, {"left": 0.25, "right": 0.75})
    time.sleep(0.03)
    environment.check_health()
    held = environment.hold(home, names)
    environment.close()

    assert held[names.index("openarm_left_joint1")] > 0.0
    assert all(side.closed for side in FakeSide.instances)
    assert {round(side.gripper, 2) for side in FakeSide.instances} == {0.25, 0.75}


def test_home_target_survives_stale_command_watchdog():
    FakeSide.instances.clear()
    runtime = load_embodiment("openarmv1")
    names = list(runtime.joint_names)
    target = runtime.home_q("down")
    target[names.index("openarm_right_joint4")] = 0.1
    environment = OpenArmCanEnvironment(
        OpenArmCanSettings(
            command_rate_hz=200.0,
            home_max_joint_speed_rad_s=1.0,
            home_timeout_s=1.0,
            home_tolerance_rad=0.001,
            watchdog_timeout_s=0.02,
        ),
        side_factory=FakeSide,
        active_sides=("right",),
    )

    environment.connect()
    try:
        environment.home(target, names)
        np.testing.assert_allclose(
            FakeSide.instances[0].q[3], 0.1, rtol=0.0, atol=0.001
        )
    finally:
        environment.close()


def test_home_commands_closed_gripper_before_live_feetech_control():
    FakeSide.instances.clear()
    runtime = load_embodiment("openarmv1")
    environment = OpenArmCanEnvironment(
        OpenArmCanSettings(
            command_rate_hz=200.0,
            home_max_joint_speed_rad_s=100.0,
        ),
        side_factory=FakeSide,
        active_sides=("right",),
    )

    environment.connect()
    try:
        environment.home(runtime.home_q("down"), list(runtime.joint_names))
        assert FakeSide.instances[0].gripper == 0.0
    finally:
        environment.close()


def test_forward_open_home_spreads_shoulders_before_bending_elbows():
    runtime = load_embodiment("openarmv1")
    names = list(runtime.joint_names)
    environment = OpenArmCanEnvironment(OpenArmCanSettings())
    environment.streamer = mock.Mock()
    measured = {
        "left": np.zeros(7, dtype=np.float32),
        "right": np.zeros(7, dtype=np.float32),
    }
    environment.streamer.feedback.return_value = measured

    environment.move_home(runtime.home_q("forward_open"), names)

    calls = environment.streamer.set_targets.call_args_list
    clearance, clearance_grippers = calls[0].args
    final, final_grippers = calls[1].args
    np.testing.assert_allclose(clearance["left"][:3], [0.0, -np.pi / 9, np.pi / 18])
    np.testing.assert_allclose(clearance["right"][:3], [0.0, np.pi / 9, -np.pi / 18])
    np.testing.assert_allclose(clearance["left"][3:], 0.0)
    np.testing.assert_allclose(clearance["right"][3:], 0.0)
    np.testing.assert_allclose(final["left"][3], np.pi / 2)
    np.testing.assert_allclose(final["right"][3], np.pi / 2)
    assert clearance_grippers == final_grippers == {"left": 0.0, "right": 0.0}
    assert environment.streamer.wait_until_targets.call_count == 2


def test_invalid_target_shape_is_rejected():
    FakeSide.instances.clear()
    environment = OpenArmCanEnvironment(
        OpenArmCanSettings(command_rate_hz=100.0), side_factory=FakeSide
    )
    runtime = load_embodiment("openarmv1")
    environment.connect()
    environment.home(runtime.home_q(), list(runtime.joint_names))
    try:
        try:
            environment.streamer.set_targets({"left": np.zeros(3)})  # type: ignore[union-attr]
        except ValueError as exc:
            assert "Invalid OpenArm target" in str(exc)
        else:
            raise AssertionError("invalid target was accepted")
    finally:
        environment.close()


def test_single_side_connects_and_disables_only_selected_arm():
    FakeSide.instances.clear()
    runtime = load_embodiment("openarmv1")
    environment = OpenArmCanEnvironment(
        OpenArmCanSettings(
            command_rate_hz=500.0,
            max_joint_speed_rad_s=100.0,
            home_max_joint_speed_rad_s=100.0,
        ),
        side_factory=FakeSide,
        active_sides=("right",),
    )

    environment.connect()
    environment.home(runtime.home_q(), list(runtime.joint_names))
    environment.close()

    assert [side.port for side in FakeSide.instances] == ["can0"]
    assert FakeSide.instances[0].closed


def test_urdf_joint_limit_violation_is_rejected_before_streaming():
    FakeSide.instances.clear()
    runtime = load_embodiment("openarmv1")
    names = list(runtime.joint_names)
    environment = OpenArmCanEnvironment(
        OpenArmCanSettings(),
        side_factory=FakeSide,
        active_sides=("left",),
        joint_limits={"openarm_left_joint4": (0.0, 2.443461)},
    )
    invalid = runtime.home_q()
    invalid[names.index("openarm_left_joint4")] = -0.1

    with np.testing.assert_raises_regex(ValueError, "outside URDF limits"):
        environment._split_q(invalid, names)


def test_tiny_ik_overshoot_is_snapped_to_urdf_joint_limit():
    runtime = load_embodiment("openarmv1")
    names = list(runtime.joint_names)
    joint = "openarm_left_joint5"
    lower = -1.570796
    environment = OpenArmCanEnvironment(
        OpenArmCanSettings(),
        side_factory=FakeSide,
        active_sides=("left",),
        joint_limits={joint: (lower, 1.570796)},
    )
    target = runtime.home_q()
    target[names.index(joint)] = lower - 5e-5

    split = environment._split_q(target, names)

    assert split["left"][4] == np.float32(lower)


def test_live_limit_violation_holds_only_unsafe_arm_and_recovers():
    runtime = load_embodiment("openarmv1")
    names = list(runtime.joint_names)
    joint = "openarm_left_joint5"
    environment = OpenArmCanEnvironment(
        OpenArmCanSettings(),
        side_factory=FakeSide,
        joint_limits={joint: (-1.0, 1.0)},
    )
    environment.streamer = mock.Mock()
    invalid = runtime.home_q()
    invalid[names.index(joint)] = -1.1

    environment.command(invalid, names, {"left": 0.25, "right": 0.75})

    targets, grippers = environment.streamer.set_targets.call_args.args
    assert set(targets) == {"right"}
    assert grippers == {"left": 0.25, "right": 0.75}

    recovered = invalid.copy()
    recovered[names.index(joint)] = -0.9
    environment.command(recovered, names, {"left": 0.25, "right": 0.75})

    targets, _ = environment.streamer.set_targets.call_args.args
    assert set(targets) == {"left", "right"}
    assert targets["left"][4] == np.float32(-0.9)
