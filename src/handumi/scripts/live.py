#!/usr/bin/env python3
"""Backward-compatible alias for :mod:`handumi.scripts.teleop_sim`."""

from __future__ import annotations

from handumi.scripts.teleop_sim import *  # noqa: F401,F403
from handumi.scripts.teleop_sim import main


if __name__ == "__main__":
    main()
