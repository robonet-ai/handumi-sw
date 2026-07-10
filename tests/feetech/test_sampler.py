import time

from handumi.feetech import FeetechGripperSampler, GripperWidths


class _Pair:
    def __init__(self):
        self.value = 0

    def read_normalized_widths(self):
        self.value += 1
        return GripperWidths(
            left=self.value / 1000.0,
            right=self.value / 1000.0,
            left_mm=float(self.value),
            right_mm=float(self.value),
            left_normalized=0.1,
            right_normalized=0.1,
            left_ticks=self.value,
            right_ticks=self.value,
        )


def test_sampler_keeps_native_rate_history():
    sampler = FeetechGripperSampler(_Pair(), sample_hz=200.0)
    sampler.start()
    try:
        time.sleep(0.03)
        latest = sampler.latest()
        assert latest is not None
        assert latest.sequence >= 3
        selected = sampler.sample_at(latest.sample_time_ns)
        assert selected == latest
        assert sampler.consecutive_errors == 0
    finally:
        sampler.stop()
