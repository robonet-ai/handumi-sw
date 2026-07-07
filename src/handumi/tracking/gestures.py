"""Hands-free gesture detection from Feetech gripper widths.

Shared by any script that needs a hands-free control signal while wearing the
HandUMI shells (no free fingers to reach a controller button): today
``handumi.scripts.live_tracking_quest`` uses it to reset the workspace, and
``handumi.scripts.record_handumi_quest`` uses it to start/stop an episode.
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
        close_mm: float = 8.0,
        open_mm: float = 25.0,
        window_s: float = 1.2,
    ) -> None:
        self._close_mm = close_mm
        self._open_mm = open_mm
        self._window_s = window_s
        self._armed = {"left": True, "right": True}  # seen open since last clap
        self._last_clap_t: dict[str, float | None] = {"left": None, "right": None}

    def update(self, left_mm: float, right_mm: float, now_s: float) -> bool:
        """Feed one width sample; returns True when either side double-claps."""
        triggered = False
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
                    triggered = True
                else:
                    self._last_clap_t[side] = now_s
        return triggered
