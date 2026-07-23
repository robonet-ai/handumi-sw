#!/usr/bin/env python3
"""Replay one deterministic Piper motion to isolate teleop jitter.

This is an executable hardware diagnostic, not a pytest test.  It deliberately
does not use tracking or IK: both modes evaluate the same smooth, hard-coded
joint trajectory.  First inspect it in Viser, then replay the exact same
trajectory on the real arms and compare the requested positions with CAN
feedback.

The trajectory starts and ends at ``piper.yaml``'s configured home pose.  It
is a manipulation-like reach/retract: shoulder and elbow carry each gripper
forward, while J4/J6 make two small orientation corrections during each 6 s
cycle.  It stays well within the limits in ``assets/piper/piper.urdf``.

Examples::

    # Kinematic reference.  Open the printed Viser URL.
    .venv/bin/python tests/real/piper/test_jitter.py --mode sim

    # Same commands on the real robot.  This homes both arms first and returns
    # them home on exit, so the explicit confirmation is required.
    .venv/bin/python tests/real/piper/test_jitter.py --mode real \
        --confirm "RUN PIPER JITTER TEST"

    # Save target, streamed-command, and feedback samples for later plotting.
    .venv/bin/python tests/real/piper/test_jitter.py --mode real \\
        --confirm "RUN PIPER JITTER TEST" --csv /tmp/piper-jitter.csv

Interpretation:

* Smooth ``sim`` but a noisy/erroring real feedback trace points to the
  real command path, CAN timing, motor control, or mechanics -- not tracking
  or IK.
* A non-smooth target trace in either mode means the commanded source needs
  attention before investigating the mechanism.
* A smooth target and feedback with visible vibration is mechanical (mount,
  backlash, resonance, cabling, or payload) rather than command jitter.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from handumi.config import DEFAULT_RIG_CONFIG
from handumi.real.can_setup import ensure_can_interfaces_ready
from handumi.real.piper.driver import (
    PiperCanEnvironment,
    load_piper_can_settings,
    piper_mdeg_to_q,
    q_to_piper_mdeg,
)
from handumi.robots.registry import load_embodiment, resolve_home_q


LOG = logging.getLogger("handumi.piper_jitter")
CONFIRMATION = "RUN PIPER JITTER TEST"
SIDES = ("left", "right")


@dataclass(frozen=True)
class Sample:
    """One requested trajectory sample in full URDF joint order."""

    elapsed_s: float
    q: np.ndarray


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("sim", "real"),
        required=True,
        help="Run the visual kinematic reference or send it to both real Piper arms.",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=24.0,
        help="Replay duration in seconds; default is four complete 6 s cycles.",
    )
    parser.add_argument(
        "--trajectory-period-s",
        type=float,
        default=6.0,
        help="Duration of one home → forward reach → home cycle.",
    )
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=100.0,
        help="Target update cadence in both modes; match the Piper default (100 Hz).",
    )
    parser.add_argument("--port", type=int, default=8004, help="Viser port in sim mode.")
    parser.add_argument("--no-browser", action="store_true", help="Do not open Viser automatically.")
    parser.add_argument("--no-viser", action="store_true", help="Run sim without the Viser view.")
    parser.add_argument(
        "--rig-config",
        type=Path,
        default=DEFAULT_RIG_CONFIG,
        help="Machine-local Piper CAN configuration (real mode only).",
    )
    parser.add_argument(
        "--repair-can",
        action="store_true",
        help="Allow CAN repair with sudo if the configured interfaces are not ready.",
    )
    parser.add_argument(
        "--confirm",
        default="",
        help=f'Required in real mode: --confirm "{CONFIRMATION}".',
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV output. Real mode includes command and feedback for each joint.",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.duration_s <= 0.0:
        raise SystemExit("--duration-s must be > 0.")
    if args.trajectory_period_s <= 0.0:
        raise SystemExit("--trajectory-period-s must be > 0.")
    if args.rate_hz <= 0.0:
        raise SystemExit("--rate-hz must be > 0.")
    if args.mode == "real" and args.confirm != CONFIRMATION:
        raise SystemExit(
            "Real motion is disabled. Re-run with "
            f'--confirm "{CONFIRMATION}" after clearing the workspace.'
        )


def hardcoded_sweep_q(
    home_q: np.ndarray,
    joint_names: tuple[str, ...],
    elapsed_s: float,
    period_s: float,
) -> np.ndarray:
    """Return the fixed, smooth diagnostic sweep at ``elapsed_s``.

    The reach is zero at home and one at the farthest point, with zero velocity
    at both ends.  A smaller wrist waveform gives two orientation corrections
    per reach, as when keeping a gripper level while teleoperating.  This is
    faster and more representative than a single, slow all-joint sweep, but
    still avoids position steps. Values are radians and remain within the
    Piper URDF limits.
    """
    reach = 0.5 * (1.0 - math.cos(2.0 * math.pi * elapsed_s / period_s))
    wrist_correction = math.sin(4.0 * math.pi * elapsed_s / period_s)
    q = np.asarray(home_q, dtype=np.float32).copy()
    # Reach offsets from home for [J1, J2, J3, J4, J5, J6]. This coordinated
    # bend translates each TCP about 4--5 cm forward from home (and upward,
    # clearing the table) instead of only rotating the wrist. J2 moves only
    # positive and J3 only negative, respecting their asymmetric URDF limits.
    offsets_deg = {
        "left": (15.0, 30.0, -40.0, 20.0, -10.0, 25.0),
        "right": (-15.0, 30.0, -40.0, -20.0, -10.0, -25.0),
    }
    for side, offsets in offsets_deg.items():
        for number, offset_deg in enumerate(offsets, start=1):
            index = joint_names.index(f"{side}_joint{number}")
            q[index] += np.deg2rad(offset_deg) * reach
        # Keep orientation actively changing while the arm is reaching. The
        # sign is mirrored so both grippers perform the same task-like motion.
        sign = 1.0 if side == "left" else -1.0
        q[joint_names.index(f"{side}_joint4")] += np.deg2rad(7.0 * sign) * wrist_correction
        q[joint_names.index(f"{side}_joint6")] += np.deg2rad(9.0 * sign) * wrist_correction
    return q


def trajectory_samples(
    home_q: np.ndarray,
    joint_names: tuple[str, ...],
    *,
    duration_s: float,
    period_s: float,
    rate_hz: float,
) -> Iterator[Sample]:
    """Yield a fixed-rate version of the same target used in each mode."""
    count = int(math.floor(duration_s * rate_hz)) + 1
    for index in range(count):
        elapsed_s = min(index / rate_hz, duration_s)
        yield Sample(
            elapsed_s=elapsed_s,
            q=hardcoded_sweep_q(home_q, joint_names, elapsed_s, period_s),
        )


def _open_viser(runtime, home_q: np.ndarray, args: argparse.Namespace):
    if args.no_viser:
        return None, None
    try:
        import viser
        from viser.extras import ViserUrdf
    except ModuleNotFoundError as exc:
        raise SystemExit("Simulation needs Viser. Install it with: uv sync --extra sim") from exc

    server = viser.ViserServer(port=args.port)
    server.scene.add_grid("/grid", width=3.0, height=3.0, cell_size=0.1)
    robot_view = ViserUrdf(server, runtime.load_urdf(load_meshes=True), root_node_name="/robot")
    robot_view.update_cfg(home_q)
    url = f"http://localhost:{server.get_port()}"
    LOG.info("Viser reference view: %s", url)
    if not args.no_browser:
        webbrowser.open(url)
    return server, robot_view


def _csv_writer(path: Path | None, joint_names: tuple[str, ...]):
    if path is None:
        return None, None
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", newline="", encoding="utf-8")
    columns = ["elapsed_s", "actual_elapsed_s", "schedule_lateness_ms"]
    for prefix in ("target_rad", "command_rad", "feedback_rad"):
        columns.extend(f"{prefix}_{name}" for name in joint_names)
    writer = csv.DictWriter(handle, fieldnames=columns)
    writer.writeheader()
    return handle, writer


def _write_csv_sample(
    writer,
    sample: Sample,
    joint_names,
    *,
    actual_elapsed_s: float,
    command_q=None,
    feedback_q=None,
) -> None:
    if writer is None:
        return
    row = {
        "elapsed_s": f"{sample.elapsed_s:.6f}",
        "actual_elapsed_s": f"{actual_elapsed_s:.6f}",
        "schedule_lateness_ms": f"{(actual_elapsed_s - sample.elapsed_s) * 1000.0:.3f}",
    }
    for prefix, q in (
        ("target_rad", sample.q),
        ("command_rad", command_q),
        ("feedback_rad", feedback_q),
    ):
        if q is None:
            continue
        row.update({f"{prefix}_{name}": float(q[i]) for i, name in enumerate(joint_names)})
    writer.writerow(row)


def _run_sim(runtime, home_q: np.ndarray, args: argparse.Namespace) -> None:
    _, robot_view = _open_viser(runtime, home_q, args)
    handle, writer = _csv_writer(args.csv, runtime.joint_names)
    try:
        started = time.perf_counter()
        for sample in trajectory_samples(
            home_q,
            runtime.joint_names,
            duration_s=args.duration_s,
            period_s=args.trajectory_period_s,
            rate_hz=args.rate_hz,
        ):
            actual_elapsed_s = time.perf_counter() - started
            if robot_view is not None:
                robot_view.update_cfg(sample.q)
            _write_csv_sample(
                writer,
                sample,
                runtime.joint_names,
                actual_elapsed_s=actual_elapsed_s,
                command_q=sample.q,
            )
            _sleep_to_sample(started, sample.elapsed_s)
    finally:
        if handle is not None:
            handle.close()


def _sleep_to_sample(started: float, elapsed_s: float) -> None:
    remaining = started + elapsed_s - time.perf_counter()
    if remaining > 0.0:
        time.sleep(remaining)


def _run_real(runtime, home_q: np.ndarray, args: argparse.Namespace) -> None:
    settings = load_piper_can_settings(args.rig_config, runtime.config.real)
    ensure_can_interfaces_ready(
        [settings.left_port, settings.right_port],
        bitrate=settings.bitrate,
        restart_ms=settings.restart_ms,
        repair=args.repair_can,
    )
    environment = PiperCanEnvironment(settings)
    handle, writer = _csv_writer(args.csv, runtime.joint_names)
    errors_deg: list[float] = []
    target_to_command_errors_deg: list[float] = []
    homed = False
    returned_home = False
    try:
        environment.connect()
        LOG.info("Moving both Piper arms to the configured home pose before replay.")
        environment.home(q_to_piper_mdeg(home_q, runtime.joint_names))
        homed = True
        started = time.perf_counter()
        arm_indices = [
            runtime.joint_names.index(f"{side}_joint{number}")
            for side in SIDES
            for number in range(1, 7)
        ]
        for sample in trajectory_samples(
            home_q,
            runtime.joint_names,
            duration_s=args.duration_s,
            period_s=args.trajectory_period_s,
            rate_hz=args.rate_hz,
        ):
            actual_elapsed_s = time.perf_counter() - started
            target_mdeg = q_to_piper_mdeg(sample.q, runtime.joint_names)
            environment.set_targets(target_mdeg)
            commands = environment.latest_commands_mdeg()
            feedback = environment.feedback_mdeg()
            command_q = piper_mdeg_to_q(
                left_mdeg=commands["left"],
                right_mdeg=commands["right"],
                actuated_names=runtime.joint_names,
                base_q=home_q,
            )
            feedback_q = piper_mdeg_to_q(
                left_mdeg=feedback["left"],
                right_mdeg=feedback["right"],
                actuated_names=runtime.joint_names,
                base_q=home_q,
            )
            errors_deg.append(
                float(np.max(np.abs(feedback_q[arm_indices] - command_q[arm_indices])))
                * 180.0
                / math.pi
            )
            target_to_command_errors_deg.append(
                float(np.max(np.abs(sample.q[arm_indices] - command_q[arm_indices])))
                * 180.0
                / math.pi
            )
            _write_csv_sample(
                writer,
                sample,
                runtime.joint_names,
                actual_elapsed_s=actual_elapsed_s,
                command_q=command_q,
                feedback_q=feedback_q,
            )
            _sleep_to_sample(started, sample.elapsed_s)
        LOG.info(
            "Target minus streamed command: max=%.3f deg, RMS=%.3f deg.",
            max(target_to_command_errors_deg, default=0.0),
            float(np.sqrt(np.mean(np.square(target_to_command_errors_deg))))
            if target_to_command_errors_deg
            else 0.0,
        )
        LOG.info(
            "Feedback minus streamed command: max=%.3f deg, RMS=%.3f deg.",
            max(errors_deg, default=0.0),
            float(np.sqrt(np.mean(np.square(errors_deg)))) if errors_deg else 0.0,
        )
        LOG.info("Replay finished; returning both arms home slowly.")
        environment.move_home(q_to_piper_mdeg(home_q, runtime.joint_names))
        returned_home = True
    finally:
        try:
            if homed and not returned_home:
                LOG.warning("Replay interrupted; returning both arms home slowly.")
                try:
                    environment.move_home(q_to_piper_mdeg(home_q, runtime.joint_names))
                except Exception as exc:  # pragma: no cover - real hardware recovery
                    LOG.error("Could not return Piper home during cleanup: %s", exc)
            environment.close()
        finally:
            if handle is not None:
                handle.close()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    runtime = load_embodiment("piper")
    _, home_q = resolve_home_q(runtime, rig_config=args.rig_config)
    LOG.info(
        "Mode=%s, trajectory=%.1f s cycle, replay=%.1f s, target rate=%.1f Hz.",
        args.mode,
        args.trajectory_period_s,
        args.duration_s,
        args.rate_hz,
    )
    try:
        if args.mode == "sim":
            _run_sim(runtime, home_q, args)
        else:
            _run_real(runtime, home_q, args)
    except KeyboardInterrupt:
        LOG.info("Stopped by user.")


if __name__ == "__main__":
    main()
