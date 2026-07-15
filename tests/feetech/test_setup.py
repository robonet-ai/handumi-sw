import tempfile
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import yaml

from handumi.feetech.setup import (
    ensure_feetech_serial_permissions,
    identify_feetech_by_replug,
    list_feetech_serial_ports,
    run_feetech_wizard,
)


def _completed(cmd, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr=stderr)


class FeetechSerialPortsTest(unittest.TestCase):
    def test_lists_usb_and_acm_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ttyUSB0").touch()
            (root / "ttyACM1").touch()
            (root / "ttyS0").touch()

            ports = list_feetech_serial_ports(
                patterns=(str(root / "ttyUSB*"), str(root / "ttyACM*"))
            )

        self.assertEqual(ports, {str(root / "ttyUSB0"), str(root / "ttyACM1")})


class FeetechWizardTest(unittest.TestCase):
    def test_preflight_adds_user_to_missing_serial_group_and_aborts_for_relogin(self):
        calls = []
        fake_stat = SimpleNamespace(st_gid=986)
        fake_group = SimpleNamespace(gr_name="dialout")

        def runner(cmd, **kwargs):
            calls.append(cmd)
            return _completed(cmd)

        with (
            mock.patch("handumi.feetech.setup.os.access", return_value=False),
            mock.patch("handumi.feetech.setup.os.stat", return_value=fake_stat),
            mock.patch("handumi.feetech.setup.grp.getgrgid", return_value=fake_group),
            mock.patch("handumi.feetech.setup.os.getgroups", return_value=[]),
            self.assertRaisesRegex(SystemExit, "Permisos seriales actualizados"),
        ):
            ensure_feetech_serial_permissions(
                list_ports_fn=lambda: {"/dev/ttyACM0"},
                runner=runner,
                user="raul",
            )

        self.assertIn(["sudo", "-v"], calls)
        self.assertIn(["sudo", "usermod", "-aG", "dialout", "raul"], calls)

    def test_preflight_does_nothing_when_ports_are_accessible(self):
        calls = []
        with mock.patch("handumi.feetech.setup.os.access", return_value=True):
            ensure_feetech_serial_permissions(
                list_ports_fn=lambda: {"/dev/ttyACM0"},
                runner=lambda cmd, **kwargs: calls.append(cmd) or _completed(cmd),
            )

        self.assertEqual(calls, [])

    def test_identifies_port_added_after_prompt_and_scans_id(self):
        ports: set[str] = set()
        prompts: list[str] = []

        def input_fn(prompt: str) -> str:
            prompts.append(prompt)
            if "Conecta" in prompt:
                ports.add("/dev/ttyUSB0")
            return ""

        with mock.patch("handumi.feetech.setup.os.access", return_value=True):
            ref = identify_feetech_by_replug(
                "derecho",
                start_id=0,
                end_id=20,
                baudrate=1_000_000,
                protocol_version=0,
                input_fn=input_fn,
                list_ports_fn=lambda: set(ports),
                scan_ids_fn=lambda port: [7],
                poll_s=0.001,
            )

        self.assertEqual(ref.port, "/dev/ttyUSB0")
        self.assertEqual(ref.servo_id, 7)
        self.assertEqual(len(prompts), 2)

    def test_rejects_multiple_ids_on_one_adapter(self):
        ports: set[str] = set()

        def input_fn(prompt: str) -> str:
            if "Conecta" in prompt:
                ports.add("/dev/ttyUSB0")
            return ""

        with (
            mock.patch("handumi.feetech.setup.os.access", return_value=True),
            self.assertRaisesRegex(SystemExit, "multiples Feetech IDs"),
        ):
            identify_feetech_by_replug(
                "izquierdo",
                start_id=0,
                end_id=20,
                baudrate=1_000_000,
                protocol_version=0,
                input_fn=input_fn,
                list_ports_fn=lambda: set(ports),
                scan_ids_fn=lambda port: [0, 1],
                poll_s=0.001,
            )

    def test_permission_denied_reports_actionable_group_hint(self):
        ports: set[str] = set()
        fake_stat = SimpleNamespace(st_gid=986)
        fake_group = SimpleNamespace(gr_name="dialout")

        def input_fn(prompt: str) -> str:
            if "Conecta" in prompt:
                ports.add("/dev/ttyACM0")
            return ""

        with (
            mock.patch("handumi.feetech.setup.os.access", return_value=False),
            mock.patch("handumi.feetech.setup.os.stat", return_value=fake_stat),
            mock.patch("handumi.feetech.setup.grp.getgrgid", return_value=fake_group),
            mock.patch("handumi.feetech.setup.os.getgroups", return_value=[]),
            self.assertRaisesRegex(SystemExit, "sudo usermod -aG dialout"),
        ):
            identify_feetech_by_replug(
                "derecho",
                start_id=0,
                end_id=20,
                baudrate=1_000_000,
                protocol_version=0,
                input_fn=input_fn,
                list_ports_fn=lambda: set(ports),
                scan_ids_fn=lambda port: [1],
                poll_s=0.001,
            )

    def test_timeout_reports_ports_before_and_after(self):
        with self.assertRaisesRegex(SystemExit, "Puertos antes: /dev/ttyACM0"):
            identify_feetech_by_replug(
                "izquierdo",
                start_id=0,
                end_id=20,
                baudrate=1_000_000,
                protocol_version=0,
                timeout_s=0.0,
                input_fn=lambda prompt: "",
                list_ports_fn=lambda: {"/dev/ttyACM0"},
                scan_ids_fn=lambda port: [0],
                poll_s=0.001,
            )

    def test_uses_existing_unassigned_port_for_second_side(self):
        ports = {"/dev/ttyACM0", "/dev/ttyACM1"}

        with mock.patch("handumi.feetech.setup.os.access", return_value=True):
            ref = identify_feetech_by_replug(
                "izquierdo",
                start_id=0,
                end_id=20,
                baudrate=1_000_000,
                protocol_version=0,
                input_fn=lambda prompt: "",
                list_ports_fn=lambda: set(ports),
                scan_ids_fn=lambda port: [0],
                used_ports={"/dev/ttyACM0"},
                poll_s=0.001,
            )

        self.assertEqual(ref.port, "/dev/ttyACM1")
        self.assertEqual(ref.servo_id, 0)

    def test_wizard_saves_right_then_left_mapping_to_rig(self):
        with tempfile.TemporaryDirectory() as tmp:
            rig = Path(tmp) / "rig.yaml"
            rig.write_text("cameras: {}\n", encoding="utf-8")
            ports: set[str] = set()
            connect_count = 0

            def input_fn(prompt: str) -> str:
                nonlocal connect_count
                if "Conecta" in prompt:
                    port = "/dev/ttyUSB1" if connect_count == 0 else "/dev/ttyUSB0"
                    ports.add(port)
                    connect_count += 1
                return ""

            def scan_ids_fn(port: str) -> list[int]:
                return [1] if port == "/dev/ttyUSB1" else [0]

            with mock.patch("handumi.feetech.setup.os.access", return_value=True):
                left, right = run_feetech_wizard(
                    rig_config=rig,
                    start_id=0,
                    end_id=20,
                    input_fn=input_fn,
                    list_ports_fn=lambda: set(ports),
                    scan_ids_fn=scan_ids_fn,
                    poll_s=0.001,
                )
            data = yaml.safe_load(rig.read_text(encoding="utf-8"))

        self.assertEqual(left.port, "/dev/ttyUSB0")
        self.assertEqual(left.servo_id, 0)
        self.assertEqual(right.port, "/dev/ttyUSB1")
        self.assertEqual(right.servo_id, 1)
        self.assertEqual(data["feetech"]["left"]["port"], "/dev/ttyUSB0")
        self.assertEqual(data["feetech"]["left"]["servo_id"], 0)
        self.assertEqual(data["feetech"]["right"]["port"], "/dev/ttyUSB1")
        self.assertEqual(data["feetech"]["right"]["servo_id"], 1)


if __name__ == "__main__":
    unittest.main()
