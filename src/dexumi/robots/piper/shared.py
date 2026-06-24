"""Shared constants and utilities for the Piper embodiment."""

from pathlib import Path

ARM_JOINT_COUNT = 6
GRIPPER_OPEN_WIDTH_M = 0.035

# Shape (8,): 6 arm joints in radians, one unused slot, then gripper in [0, 1].
COMMAND_SIZE = 8
GRIPPER_INDEX = 7


def _resolve_urdf_path() -> Path:
    here = Path(__file__).resolve()
    candidates = (
        here.parents[2] / "assets" / "piper" / "piper.urdf",
        here.parents[4] / "assets" / "piper" / "piper.urdf",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Could not find piper.urdf; expected it under dexumi/assets/piper or repo assets/piper"
    )


URDF_PATH: Path = _resolve_urdf_path()


def urdf_arm_joint_names(*, is_left: bool) -> list[str]:
    """URDF actuated joint names for one arm, in control order (joint1..joint8)."""
    prefix = "izq" if is_left else "der"
    return [f"{prefix}_joint{i}" for i in range(1, 9)]


def gripper_to_finger_positions(gripper: float) -> tuple[float, float]:
    """Map a normalized gripper command to the two prismatic finger joints."""
    width = float(max(0.0, min(1.0, gripper))) * GRIPPER_OPEN_WIDTH_M
    return width, -width
