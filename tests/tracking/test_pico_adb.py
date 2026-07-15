import subprocess
import unittest

from handumi.tracking.pico import (
    keep_pico_awake,
    prepare_pico_adb_session,
    setup_adb_reverse,
    stop_xrt_service,
    verify_adb_connection,
)


def _completed(cmd, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr=stderr)


class PicoAdbTest(unittest.TestCase):
    def test_verify_adb_connection_detects_authorized_device(self):
        def runner(cmd, **kwargs):
            self.assertEqual(cmd, ["adb", "devices"])
            return _completed(cmd, stdout="List of devices attached\nPICO123\tdevice\n")

        self.assertTrue(verify_adb_connection(timeout_s=1.0, runner=runner))

    def test_verify_adb_connection_times_out_without_device(self):
        def runner(cmd, **kwargs):
            return _completed(cmd, stdout="List of devices attached\n")

        self.assertFalse(verify_adb_connection(timeout_s=0.0, runner=runner))

    def test_setup_adb_reverse_runs_expected_command(self):
        calls = []

        def runner(cmd, **kwargs):
            calls.append(cmd)
            return _completed(cmd)

        self.assertTrue(setup_adb_reverse(runner=runner))
        self.assertEqual(
            calls,
            [["adb", "reverse", "tcp:63901", "tcp:63901"]],
        )

    def test_keep_pico_awake_runs_stayon_and_wakeup(self):
        calls = []

        def runner(cmd, **kwargs):
            calls.append(cmd)
            return _completed(cmd)

        self.assertTrue(keep_pico_awake(runner=runner))
        self.assertEqual(
            calls,
            [
                ["adb", "shell", "svc", "power", "stayon", "usb"],
                ["adb", "shell", "input", "keyevent", "WAKEUP"],
            ],
        )

    def test_prepare_pico_adb_session_combines_check_reverse_and_awake(self):
        calls = []

        def runner(cmd, **kwargs):
            calls.append(cmd)
            if cmd == ["adb", "devices"]:
                return _completed(cmd, stdout="List of devices attached\nPICO123\tdevice\n")
            return _completed(cmd)

        self.assertTrue(prepare_pico_adb_session(runner=runner))
        self.assertIn(["adb", "devices"], calls)
        self.assertIn(["adb", "reverse", "tcp:63901", "tcp:63901"], calls)
        self.assertIn(["adb", "shell", "svc", "power", "stayon", "usb"], calls)
        self.assertIn(["adb", "shell", "input", "keyevent", "WAKEUP"], calls)

    def test_prepare_pico_adb_session_aborts_without_device(self):
        def runner(cmd, **kwargs):
            return _completed(cmd, stdout="List of devices attached\n")

        with self.assertRaises(SystemExit):
            prepare_pico_adb_session(timeout_s=0.0, runner=runner)

    def test_stop_xrt_service_cleans_known_process_patterns(self):
        calls = []

        def runner(cmd, **kwargs):
            calls.append(cmd)
            return _completed(cmd, returncode=1)

        stop_xrt_service(runner=runner)

        self.assertEqual(
            calls,
            [
                ["pkill", "-f", "/opt/apps/roboticsservice/runService.sh"],
                ["pkill", "-f", "/opt/apps/roboticsservice"],
            ],
        )


if __name__ == "__main__":
    unittest.main()
