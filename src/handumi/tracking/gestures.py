"""Hands-free gesture detection from Feetech gripper widths.

Shared by any script that needs a hands-free control signal while wearing the
HandUMI shells (no free fingers to reach a controller button): today
``handumi.scripts.teleop_sim`` uses it to reset the workspace, and
``handumi.scripts.record`` uses it to start/stop an episode (--clap-control).
"""

from __future__ import annotations


class DoubleClapDetector:
    """Squeeze either gripper shut twice in quick succession.

    Each side is tracked independently. A "clap" fires when that side's width
    drops below ``close_mm``; it must reopen past ``open_mm`` (hysteresis)
    before its next clap counts. Two claps of the *same* gripper at most
    ``window_s`` apart trigger.
    """

    def __init__(
        self,
        *,
        close_mm: float = 12.0,
        open_mm: float = 20.0,
        window_s: float = 1.6,
    ) -> None:
        # Defaults tuned on hardware (2026-07-09): the original 8/25/1.2 was
        # hard to trigger — the squeeze rarely dipped under 8mm between 30Hz
        # samples, and re-opening past 25mm within 1.2s took several tries.
        self._close_mm = close_mm
        self._open_mm = open_mm
        self._window_s = window_s
        self._armed = {"left": True, "right": True}  # seen open since last clap
        self._last_clap_t: dict[str, float | None] = {"left": None, "right": None}

    def update(self, left_mm: float, right_mm: float, now_s: float) -> bool:
        """Feed one width sample; returns True when either side double-claps."""
        return self.update_side(left_mm, right_mm, now_s) is not None

    def update_side(self, left_mm: float, right_mm: float, now_s: float) -> str | None:
        """Return the side that double-clapped, preferring right if both fire."""
        triggered: list[str] = []
        for side, mm in (("left", left_mm), ("right", right_mm)):
            if mm > self._open_mm:
                self._armed[side] = True
                last = self._last_clap_t[side]
                if last is not None and now_s - last > self._window_s:
                    self._last_clap_t[side] = None  # first clap expired
                continue
            if mm < self._close_mm and self._armed[side]:
                self._armed[side] = False
                last = self._last_clap_t[side]
                if last is not None and now_s - last <= self._window_s:
                    self._last_clap_t[side] = None
                    triggered.append(side)
                else:
                    self._last_clap_t[side] = now_s
        if "right" in triggered:
            return "right"
        return triggered[0] if triggered else None
