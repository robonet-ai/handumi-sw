import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from handumi.real.backends.setup import (
    RobotSetupOptions,
    _check_openarm_motors,
    _run_openarm_zero_calibration,
    _setup_openarm,
)


def test_openarm_motor_check_uses_read_only_parameter_queries():
    output = "\n".join(f"MOTOR ID: 0x{motor:x}" for motor in range(1, 9))
    with mock.patch(
        "handumi.real.backends.setup.subprocess.run",
        return_value=subprocess.CompletedProcess([], 0, stdout=output, stderr=""),
    ) as run:
        _check_openarm_motors("right", "can0")

    command = run.call_args.args[0]
    assert "show_param" in command
    assert "monitor" not in command
    assert command[-1] == "1,2,3,4,5,6,7,8"


def test_openarm_motor_check_rejects_any_missing_response():
    output = "\n".join(f"MOTOR ID: 0x{motor:x}" for motor in range(1, 9))
    output += "\n[!] NO RESPONSE FROM MOTOR"
    with (
        mock.patch(
            "handumi.real.backends.setup.subprocess.run",
            return_value=subprocess.CompletedProcess([], 0, stdout=output, stderr=""),
        ),
        pytest.raises(SystemExit, match="diagnostic failed"),
    ):
        _check_openarm_motors("left", "can1")


def test_openarm_zero_calibration_uses_active_python_and_installed_v1_script():
    with mock.patch("handumi.real.backends.setup.subprocess.run") as run:
        _run_openarm_zero_calibration(
            "right_arm",
            "can1",
            executable="/usr/bin/openarm-can-zero-position-calibration",
        )

    command = run.call_args.args[0]
    assert command[0].endswith("python") or "python" in command[0]
    assert command[1] == "/usr/bin/openarm-can-zero-position-calibration"
    assert command[2:] == ["--canport", "can1", "--arm-side", "right_arm"]
    assert "--robot-version" not in command
    assert run.call_args.kwargs["check"] is True


def test_openarm_setup_calibrates_only_selected_physical_side():
    options = RobotSetupOptions(
        robot="openarmv1",
        rig_config=Path("configs/rig.yaml"),
        bitrate=1_000_000,
        dbitrate=5_000_000,
        restart_ms=100,
        skip_can_map=True,
        skip_can_repair=False,
        skip_motor_check=True,
        calibrate_openarm_zero=True,
        openarm_zero_side="right",
    )
    settings = SimpleNamespace(
        left_port="can1",
        right_port="can0",
        bitrate=1_000_000,
        dbitrate=5_000_000,
    )
    executable = "/usr/bin/openarm-can-zero-position-calibration"

    with (
        mock.patch("handumi.real.backends.setup.shutil.which") as which,
        mock.patch("handumi.real.backends.setup.require_openarm_can"),
        mock.patch(
            "handumi.real.backends.setup.load_openarm_settings",
            return_value=settings,
        ),
        mock.patch("handumi.real.backends.setup.load_robot_config"),
        mock.patch("handumi.real.backends.setup.ensure_can_fd_interfaces_ready"),
        mock.patch("builtins.input", return_value="CALIBRATE RIGHT"),
        mock.patch(
            "handumi.real.backends.setup._run_openarm_zero_calibration"
        ) as calibrate,
    ):
        which.side_effect = lambda name: (
            executable if name == "openarm-can-zero-position-calibration" else name
        )
        _setup_openarm(options)

    calibrate.assert_called_once_with(
        "right_arm", "can0", executable=executable
    )
