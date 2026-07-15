import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

from handumi.real.can_setup import (
    can_ready,
    ensure_can_interfaces_ready,
    identify_can_by_replug,
    read_can_status,
    run_piper_can_wizard,
)


def _completed(cmd, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr=stderr)


class CanStatusTest(unittest.TestCase):
    def test_reads_ip_link_details(self):
        text = """22: can0: <NOARP,UP,LOWER_UP,ECHO> mtu 16 qdisc pfifo_fast state UP mode DEFAULT group default qlen 10
    link/can
    can state ERROR-ACTIVE restart-ms 100
          bitrate 1000000 sample-point 0.750
"""

        status = read_can_status("can0", runner=lambda *a, **kw: _completed(a[0], stdout=text))

        self.assertTrue(status.exists)
        self.assertTrue(status.up)
        self.assertTrue(status.lower_up)
        self.assertEqual(status.state, "ERROR-ACTIVE")
        self.assertEqual(status.bitrate, 1_000_000)
        self.assertTrue(can_ready(status, bitrate=1_000_000))

    def test_missing_interface_is_not_ready(self):
        status = read_can_status(
            "can9",
            runner=lambda *a, **kw: _completed(a[0], returncode=1, stderr="missing"),
        )

        self.assertFalse(status.exists)
        self.assertFalse(can_ready(status, bitrate=1_000_000))


class CanRepairTest(unittest.TestCase):
    def test_repairs_not_ready_interface_with_explicit_sudo(self):
        calls = []
        show_count = 0

        def runner(cmd, **kwargs):
            nonlocal show_count
            calls.append(cmd)
            if cmd[:4] == ["ip", "-details", "link", "show"]:
                show_count += 1
                if show_count == 1:
                    return _completed(
                        cmd,
                        stdout=(
                            "1: can0: <NOARP,ECHO> mtu 16\n"
                            "    can state BUS-OFF restart-ms 0\n"
                            "          bitrate 500000\n"
                        ),
                    )
                return _completed(
                    cmd,
                    stdout=(
                        "1: can0: <NOARP,UP,LOWER_UP,ECHO> mtu 16\n"
                        "    can state ERROR-ACTIVE restart-ms 100\n"
                        "          bitrate 1000000\n"
                    ),
                )
            return _completed(cmd)

        ensure_can_interfaces_ready(["can0"], bitrate=1_000_000, restart_ms=100, runner=runner)

        self.assertIn(["sudo", "-v"], calls)
        self.assertIn(["sudo", "ip", "link", "set", "can0", "down"], calls)
        self.assertIn(
            [
                "sudo",
                "ip",
                "link",
                "set",
                "can0",
                "type",
                "can",
                "bitrate",
                "1000000",
                "restart-ms",
                "100",
            ],
            calls,
        )
        self.assertIn(["sudo", "ip", "link", "set", "can0", "up"], calls)

    def test_repairs_with_fallback_when_restart_ms_is_unsupported(self):
        calls = []

        def runner(cmd, **kwargs):
            calls.append(cmd)
            if cmd[:4] == ["ip", "-details", "link", "show"]:
                return _completed(
                    cmd,
                    stdout=(
                        "1: can0: <NOARP,UP,LOWER_UP,ECHO> mtu 16\n"
                        "    can state ERROR-ACTIVE restart-ms 0\n"
                        "          bitrate 1000000\n"
                    ),
                )
            if "restart-ms" in cmd:
                return _completed(
                    cmd,
                    returncode=2,
                    stderr="Error: Device doesn't support restart from Bus Off.",
                )
            return _completed(cmd)

        from handumi.real.can_setup import repair_can_interface

        repair_can_interface("can0", bitrate=1_000_000, restart_ms=100, runner=runner)

        self.assertIn(
            [
                "sudo",
                "ip",
                "link",
                "set",
                "can0",
                "type",
                "can",
                "bitrate",
                "1000000",
                "restart-ms",
                "100",
            ],
            calls,
        )
        self.assertIn(
            [
                "sudo",
                "ip",
                "link",
                "set",
                "can0",
                "type",
                "can",
                "bitrate",
                "1000000",
            ],
            calls,
        )
        self.assertIn(["sudo", "ip", "link", "set", "can0", "up"], calls)

    def test_cancelled_sudo_aborts_before_repair(self):
        def runner(cmd, **kwargs):
            if cmd[:4] == ["ip", "-details", "link", "show"]:
                return _completed(cmd, stdout="1: can0: <NOARP,ECHO> mtu 16\n")
            if cmd == ["sudo", "-v"]:
                return _completed(cmd, returncode=1)
            raise AssertionError(f"unexpected command after sudo failure: {cmd}")

        with self.assertRaises(SystemExit):
            ensure_can_interfaces_ready(
                ["can0"],
                bitrate=1_000_000,
                restart_ms=100,
                runner=runner,
            )


class CanWizardTest(unittest.TestCase):
    def test_identifies_can_added_after_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            sysfs = Path(tmp)
            prompts = []

            def input_fn(prompt):
                prompts.append(prompt)
                if "Conecta" in prompt:
                    (sysfs / "can1").mkdir()
                return ""

            ref = identify_can_by_replug(
                "derecho",
                sys_class_net=sysfs,
                input_fn=input_fn,
                poll_s=0.001,
            )

        self.assertEqual(ref.name, "can1")
        self.assertEqual(len(prompts), 2)

    def test_wizard_saves_right_then_left_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sysfs = root / "net"
            sysfs.mkdir()
            rig = root / "rig.yaml"
            rig.write_text("robots: {}\n", encoding="utf-8")
            connect_count = 0

            def input_fn(prompt):
                nonlocal connect_count
                if "Conecta" in prompt:
                    name = "can1" if connect_count == 0 else "can0"
                    (sysfs / name).mkdir()
                    connect_count += 1
                return ""

            left, right = run_piper_can_wizard(
                rig_config=rig,
                bitrate=1_000_000,
                restart_ms=100,
                sys_class_net=sysfs,
                input_fn=input_fn,
                poll_s=0.001,
            )
            data = yaml.safe_load(rig.read_text())

        self.assertEqual(left, "can0")
        self.assertEqual(right, "can1")
        self.assertEqual(data["robots"]["piper"]["can"]["left_port"], "can0")
        self.assertEqual(data["robots"]["piper"]["can"]["right_port"], "can1")
        self.assertEqual(data["robots"]["piper"]["can"]["restart_ms"], 100)


if __name__ == "__main__":
    unittest.main()
