#!/usr/bin/env python3
"""Small command router for the common HandUMI workflow."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Command:
    module: str
    description: str


COMMANDS = {
    ("doctor",): Command("handumi.scripts.doctor", "Check recording readiness"),
    ("setup",): Command("handumi.scripts.setup.setup_hardware", "Run guided setup"),
    ("setup", "ports"): Command(
        "handumi.scripts.setup.setup_ports", "Identify cameras and serial ports"
    ),
    ("record",): Command("handumi.scripts.record", "Record demonstrations"),
    ("validate",): Command("handumi.scripts.validate", "Validate recorded episodes"),
    ("replay",): Command(
        "handumi.scripts.replay.replay_in_sim", "Replay a dataset in simulation"
    ),
    ("convert",): Command("handumi.scripts.conversion", "Convert data to a robot"),
    ("completion",): Command(
        "handumi.scripts.completion", "Enable Bash, Zsh, or Fish completion"
    ),
    ("teleop", "sim"): Command("handumi.scripts.teleop_sim", "Teleoperate in simulation"),
    ("teleop", "real"): Command("handumi.scripts.teleop_real", "Teleoperate a real robot"),
    ("camera", "pico"): Command("handumi.scripts.pico_camera", "Stream cameras to PICO"),
    ("calibrate", "grippers"): Command(
        "handumi.scripts.setup.calibrate_grippers", "Calibrate Feetech grippers"
    ),
    ("calibrate", "openarm-grippers"): Command(
        "handumi.scripts.setup.calibrate_openarm_grippers",
        "Calibrate OpenArm grippers",
    ),
    ("calibrate", "spatial"): Command(
        "handumi.scripts.setup.calibrate_spatial", "Calibrate cameras and workspace"
    ),
    ("calibrate", "tcp"): Command(
        "handumi.scripts.setup.calibrate_tcp_offset", "Calibrate controller-to-TCP"
    ),
    ("servo", "home"): Command("handumi.scripts.setup.home_servos", "Home Feetech servos"),
    ("servo", "set-id"): Command(
        "handumi.scripts.setup.set_servo_id", "Assign a Feetech servo ID"
    ),
    ("tracking", "pose"): Command(
        "handumi.scripts.setup.print_controller_pose", "Print live controller poses"
    ),
}


def _program_name() -> str:
    name = Path(sys.argv[0]).name
    return name if name in {"handumi", "hu"} else "handumi"


def _print_help(
    prefix: tuple[str, ...] = (), *, program: str = "handumi"
) -> None:
    label_prefix = f"{program} {' '.join(prefix)}" if prefix else program
    print(f"usage: {label_prefix} <command> [options]\n")
    print("Commands:")
    commands = {
        path: command
        for path, command in COMMANDS.items()
        if path[: len(prefix)] == prefix and len(path) > len(prefix)
    }
    labels = [" ".join(path) for path in commands]
    width = max(len(label) for label in labels)
    for (path, command), label in zip(commands.items(), labels, strict=True):
        print(f"  {program} {label:<{width}}  {command.description}")
    print(f"\nRun '{program} <command> --help' for command-specific options.")


def main(argv: list[str] | None = None) -> None:
    program = _program_name()
    values = list(sys.argv[1:] if argv is None else argv)
    if not values or values[0] in {"-h", "--help"}:
        _print_help(program=program)
        return
    group = (values[0],)
    group_exists = any(
        path[:1] == group and len(path) > 1 for path in COMMANDS
    ) and group not in COMMANDS
    if group_exists and (len(values) == 1 or values[1] in {"-h", "--help"}):
        _print_help(group, program=program)
        return
    match = next(
        (
            (path, command)
            for path, command in sorted(
                COMMANDS.items(), key=lambda item: len(item[0]), reverse=True
            )
            if tuple(values[: len(path)]) == path
        ),
        None,
    )
    if match is None:
        requested = " ".join(values[:2])
        raise SystemExit(
            f"Unknown HandUMI command {requested!r}. Run '{program} --help'."
        )
    path, command = match
    rest = values[len(path) :]
    module = importlib.import_module(command.module)
    previous_argv = sys.argv
    try:
        sys.argv = [f"{program} {' '.join(path)}", *rest]
        module.main()
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    main()
