"""Regression tests for the gripper read path used by ``handumi-teleop-sim``.

These tests intentionally do not open a serial device.  They model the exact
SDK result from the reported failure: a successful ID ping followed by failed
``Present_Position`` reads (``comm=-7``).
"""

from types import SimpleNamespace

import pytest

from handumi.feetech.bus import FeetechBus
from handumi.feetech.calibration import FeetechConfig, GripperCalibration
from handumi.feetech.gripper import FeetechGripperPair


class _PacketWithPingButNoPosition:
    """A servo that answers ping but times out on every position read."""

    def __init__(self) -> None:
        self.read_calls = 0

    def ping(self, *_args: object) -> tuple[int, int, int]:
        return (0, 0, 0)

    def read2ByteTxRx(self, *_args: object) -> tuple[int, int, int]:
        self.read_calls += 1
        # ``comm=-7, error=0`` is the result printed in the real traceback.
        return (0, -7, 0)


def _bus(packet: _PacketWithPingButNoPosition) -> FeetechBus:
    """Return an opened-in-memory bus, avoiding any physical serial access."""
    bus = FeetechBus("/dev/fake")
    bus._sdk = SimpleNamespace(COMM_SUCCESS=0)
    bus._port_handler = object()
    bus._packet_handler = packet
    return bus


def test_teleop_gripper_read_fails_after_a_successful_port_scan() -> None:
    """Reproduce the ``teleop_sim -> gripper -> bus`` failure path exactly."""
    config = FeetechConfig(
        port=None,
        baudrate=1_000_000,
        protocol_version=0,
        left=GripperCalibration(
            servo_id=1,
            port="/dev/ttyACM0",
            closed_ticks=1000,
            open_ticks=3000,
            max_width_mm=80.0,
        ),
        right=GripperCalibration(
            servo_id=6,
            port="/dev/ttyACM1",
            closed_ticks=1000,
            open_ticks=3000,
            max_width_mm=80.0,
        ),
    )
    grippers = FeetechGripperPair(config)
    left_packet = _PacketWithPingButNoPosition()
    grippers._buses = {
        "/dev/ttyACM0": _bus(left_packet),
        "/dev/ttyACM1": _bus(_PacketWithPingButNoPosition()),
    }

    # This is the operation performed by handumi-setup-ports.  It can pass
    # even when the register read that teleop needs is unreliable.
    assert grippers._buses["/dev/ttyACM0"].ping(1)

    # This is the same call made at teleop_sim.py:764.  FeetechBus retries four
    # times, hence five position transactions in total.
    with pytest.raises(
        RuntimeError,
        match=r"Failed to read Present_Position from servo 1 after 5 attempts "
        r"\(comm=-7, error=0\)\.",
    ):
        grippers.read_normalized_widths()

    assert left_packet.read_calls == 5
