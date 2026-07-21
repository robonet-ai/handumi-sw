#!/usr/bin/env python3
"""Small command router for the common HandUMI workflow."""

from __future__ import annotations

import importlib
import sys


COMMANDS = {
    "doctor": (
        "handumi.scripts.doctor",
        "Check installation and recording readiness",
    ),
    "setup": ("handumi.scripts.setup.setup_hardware", "Run guided hardware setup"),
    "record": ("handumi.scripts.record", "Record demonstrations"),
    "validate": ("handumi.scripts.validate", "Validate recorded episodes"),
    "replay": ("handumi.scripts.replay.replay_in_sim", "Replay a dataset in simulation"),
    "convert": ("handumi.scripts.conversion", "Convert data to a robot profile"),
}


def _print_help() -> None:
    print("usage: handumi <command> [options]\n")
    print("Common workflow commands:")
    width = max(len(command) for command in COMMANDS)
    for command, (_, description) in COMMANDS.items():
        print(f"  handumi {command:<{width}}  {description}")
    print("\nRun 'handumi <command> --help' for command-specific options.")


def main(argv: list[str] | None = None) -> None:
    values = list(sys.argv[1:] if argv is None else argv)
    if not values or values[0] in {"-h", "--help"}:
        _print_help()
        return
    command, *rest = values
    target = COMMANDS.get(command)
    if target is None:
        choices = ", ".join(COMMANDS)
        raise SystemExit(f"Unknown HandUMI command {command!r}. Choose from: {choices}.")
    module = importlib.import_module(target[0])
    previous_argv = sys.argv
    try:
        sys.argv = [f"handumi {command}", *rest]
        module.main()
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    main()
