import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from handumi.scripts.setup import setup_ports


class UsbSerialAdapterDetectionTest(unittest.TestCase):
    def test_detects_known_usb_serial_adapters_from_sysfs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            device = root / "1-3.1"
            device.mkdir()
            (device / "idVendor").write_text("1a86\n", encoding="utf-8")
            (device / "idProduct").write_text("55d3\n", encoding="utf-8")
            (device / "product").write_text("USB Single Serial\n", encoding="utf-8")
            (device / "serial").write_text("5A46083732\n", encoding="utf-8")

            adapters = setup_ports._detect_usb_serial_adapters(root)

        self.assertEqual(len(adapters), 1)
        self.assertEqual(adapters[0]["vendor"], "1a86")
        self.assertEqual(adapters[0]["product"], "55d3")
        self.assertEqual(adapters[0]["driver"], "ch341")
        self.assertEqual(adapters[0]["serial"], "5A46083732")


class SerialPortDiagnosticsTest(unittest.TestCase):
    def test_prints_driver_hint_when_adapter_exists_without_tty_device(self):
        adapters = [
            {
                "vendor": "1a86",
                "product": "55d3",
                "name": "QinHeng CH34x / USB Single Serial",
                "driver": "ch341",
                "serial": "5A46083732",
            }
        ]
        buf = io.StringIO()
        with (
            mock.patch.object(setup_ports.glob, "glob", return_value=[]),
            mock.patch.object(
                setup_ports, "_detect_usb_serial_adapters", return_value=adapters
            ),
            mock.patch.object(setup_ports, "_kernel_module_available", return_value=False),
            mock.patch.object(
                setup_ports,
                "_kernel_module_tree_hint",
                return_value="Kernel module tree is missing for running kernel 7.1.2.",
            ),
            contextlib.redirect_stdout(buf),
        ):
            setup_ports._print_serial_ports(range(0, 2))

        output = buf.getvalue()
        self.assertIn("Feetech serial ports", output)
        self.assertIn("USB serial adapters are connected", output)
        self.assertIn("1a86:55d3", output)
        self.assertIn("Driver hint: ch341", output)
        self.assertIn("Missing module for the running kernel: ch341", output)
        self.assertIn("sudo reboot", output)

    def test_serial_port_permission_hint_names_device_group(self):
        fake_stat = SimpleNamespace(st_gid=986)
        fake_group = SimpleNamespace(gr_name="uucp")
        with (
            mock.patch.object(setup_ports.os, "access", return_value=False),
            mock.patch.object(setup_ports.os, "stat", return_value=fake_stat),
            mock.patch.object(setup_ports.os, "getgroups", return_value=[]),
            mock.patch.object(setup_ports.grp, "getgrgid", return_value=fake_group),
        ):
            hint = setup_ports._serial_port_permission_hint("/dev/ttyUSB0")

        self.assertEqual(
            hint,
            [
                "Permission hint: add your user to the serial device group `uucp`.",
                "Run: sudo usermod -aG uucp $USER",
                "Then log out and back in.",
            ],
        )


if __name__ == "__main__":
    unittest.main()
