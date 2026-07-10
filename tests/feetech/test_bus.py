import unittest
from types import SimpleNamespace

from handumi.feetech.bus import FeetechBus


class _FakePacket:
    def __init__(self, *, reads=None, writes=None, pings=None):
        self.reads = list(reads or [])
        self.writes = list(writes or [])
        self.pings = list(pings or [])
        self.write_calls = 0

    def read2ByteTxRx(self, *_args):
        response = self.reads.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    def write1ByteTxRx(self, *_args):
        self.write_calls += 1
        response = self.writes.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    def ping(self, *_args):
        response = self.pings.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def _bus_with(packet: _FakePacket) -> FeetechBus:
    bus = FeetechBus("/dev/test")
    bus._sdk = SimpleNamespace(COMM_SUCCESS=0)
    bus._port_handler = object()
    bus._packet_handler = packet
    return bus


class FeetechBusRetryTest(unittest.TestCase):
    def test_read_position_retries_serial_io_errors(self):
        packet = _FakePacket(
            reads=[
                OSError("device reports readiness to read but returned no data"),
                (1234, 0, 0),
            ]
        )
        bus = _bus_with(packet)

        self.assertEqual(bus.read_position(1, retry_delay_s=0), 1234)

    def test_read_position_reports_last_retry_failure(self):
        packet = _FakePacket(reads=[OSError("no data"), OSError("still no data")])
        bus = _bus_with(packet)

        with self.assertRaisesRegex(RuntimeError, "OSError: still no data"):
            bus.read_position(1, retries=1, retry_delay_s=0)

    def test_write_retries_serial_io_errors(self):
        packet = _FakePacket(writes=[OSError("no data"), (0, 0)])
        bus = _bus_with(packet)

        bus._write_1_byte(1, 40, 0, "Torque_Enable", retry_delay_s=0)

        self.assertEqual(packet.write_calls, 2)

    def test_ping_treats_serial_io_errors_as_no_response(self):
        packet = _FakePacket(pings=[OSError("no data")])
        bus = _bus_with(packet)

        self.assertFalse(bus.ping(1))


if __name__ == "__main__":
    unittest.main()
