#!/usr/bin/env python3
"""CLI wrapper for :mod:`handumi.capture.live_tracking`."""

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from handumi.capture.live_tracking import main


if __name__ == "__main__":
    main()
