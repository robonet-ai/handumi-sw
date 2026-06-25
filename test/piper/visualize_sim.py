#!/usr/bin/env python3
"""Web-based Viser visualizer for the dual Piper URDF."""

from __future__ import annotations

import argparse
import asyncio
import math
import time

import numpy as np

from dexumi.robots.piper.sim import Sim


def _rest_command() -> np.ndarray:
    """Piper's standard all-zero arm pose with an open gripper."""
    command = np.zeros(8, dtype=np.float32)
    command[7] = 1.0
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Open a Viser web viewer for the dual Piper URDF."
    )
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument(
        "--demo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Oscillate the arms slowly for a quick visual check (default: on).",
    )
    parser.add_argument(
        "--hold-after",
        type=float,
        default=None,
        help="Seconds to keep Viser alive after a non-demo run. Default: indefinite.",
    )
    return parser


async def run_demo(sim: Sim, *, playback_fps: float) -> None:
    """Replay a simple periodic motion on both arms."""
    rest = _rest_command()
    left = rest.copy()
    right = rest.copy()
    await sim.motion_control(left=left, right=right)

    frame_delay = 0.0 if playback_fps <= 0 else 1.0 / playback_fps
    next_time = time.perf_counter()
    start = time.perf_counter()

    while True:
        phase = time.perf_counter() - start
        offset = 0.25 * math.sin(phase)
        left = rest.copy()
        right = rest.copy()
        left[0] = rest[0] + offset
        right[0] = rest[0] - offset
        left[7] = 0.5 + 0.5 * math.sin(phase * 0.7)
        right[7] = left[7]
        await sim.motion_control(left=left, right=right)

        next_time += frame_delay
        if frame_delay > 0:
            await asyncio.sleep(max(0.0, next_time - time.perf_counter()))


async def main_async() -> None:
    args = build_parser().parse_args()

    sim = Sim(port=args.port)
    await sim.enable()
    await asyncio.sleep(0.5)

    print(f"Viser simulation: http://localhost:{args.port}")
    if args.demo:
        print("Running demo motion. Press Ctrl+C to stop.")
        await run_demo(sim, playback_fps=30.0)
        return

    left = _rest_command()
    await sim.motion_control(left=left, right=left.copy())
    print("Simulation running. Press Ctrl+C to stop.")

    if args.hold_after is not None:
        await asyncio.sleep(args.hold_after)
    else:
        await asyncio.Event().wait()


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
