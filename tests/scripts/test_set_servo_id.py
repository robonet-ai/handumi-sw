import argparse
import contextlib
import io
import unittest
from unittest import mock

from handumi.scripts.setup import set_servo_id


class FakeBus:
    present_ids: set[int] = set()
    writes: list[tuple[int, int]] = []

    def __init__(self, *, port: str, baudrate: int, protocol_version: int) -> None:
        self.port = port
        self.baudrate = baudrate
        self.protocol_version = protocol_version

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def ping(self, servo_id: int) -> bool:
        return int(servo_id) in self.present_ids

    def scan(self, ids) -> list[int]:
        return [int(servo_id) for servo_id in ids if int(servo_id) in self.present_ids]

    def write_servo_id(self, old_id: int, new_id: int) -> None:
        self.writes.append((old_id, new_id))
        self.present_ids.discard(old_id)
        self.present_ids.add(new_id)


def _args(**overrides):
    values = {
        "port": "/dev/ttyUSB0",
        "old_id": None,
        "new_id": 2,
        "baudrate": 1_000_000,
        "protocol_version": 0,
        "scan_start_id": 0,
        "scan_end_id": 253,
        "yes": True,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class SetServoIdTest(unittest.TestCase):
    def setUp(self) -> None:
        FakeBus.present_ids = {1}
        FakeBus.writes = []

    def test_writes_new_id_when_old_replies_and_new_is_free(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            set_servo_id.set_servo_id(_args(), bus_cls=FakeBus)

        self.assertEqual(FakeBus.writes, [(1, 2)])
        self.assertEqual(FakeBus.present_ids, {2})
        self.assertIn("Servo ID updated", buf.getvalue())

    def test_noop_when_servo_already_has_requested_id(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            set_servo_id.set_servo_id(_args(new_id=1), bus_cls=FakeBus)

        self.assertEqual(FakeBus.writes, [])
        self.assertIn("already ID 1", buf.getvalue())

    def test_rejects_when_new_id_is_already_present(self):
        FakeBus.present_ids = {1, 2}
        with self.assertRaisesRegex(SystemExit, "already replies"):
            set_servo_id.set_servo_id(_args(old_id=1), bus_cls=FakeBus)
        self.assertEqual(FakeBus.writes, [])

    def test_rejects_id_outside_supported_range(self):
        with self.assertRaisesRegex(SystemExit, "must be in"):
            set_servo_id.set_servo_id(_args(new_id=254), bus_cls=FakeBus)
        self.assertEqual(FakeBus.writes, [])

    def test_requires_old_id_when_multiple_servos_are_connected(self):
        FakeBus.present_ids = {1, 3}
        with self.assertRaisesRegex(SystemExit, "Multiple servo IDs"):
            set_servo_id.set_servo_id(_args(), bus_cls=FakeBus)
        self.assertEqual(FakeBus.writes, [])

    def test_interactive_confirm_accepts_y(self):
        with mock.patch("builtins.input", return_value="y"):
            set_servo_id.set_servo_id(_args(yes=False), bus_cls=FakeBus)
        self.assertEqual(FakeBus.writes, [(1, 2)])

    def test_interactive_confirm_rejects_other_input(self):
        with (
            mock.patch("builtins.input", return_value="yes"),
            self.assertRaisesRegex(SystemExit, "Aborted"),
        ):
            set_servo_id.set_servo_id(_args(yes=False), bus_cls=FakeBus)
        self.assertEqual(FakeBus.writes, [])


if __name__ == "__main__":
    unittest.main()
