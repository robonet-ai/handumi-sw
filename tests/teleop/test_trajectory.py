import threading
import time

import numpy as np
import pytest

from handumi.teleop.trajectory import (
    DelayedJointCommandBuffer,
    DelayedJointCommandPlayer,
)


def test_delayed_buffer_interpolates_at_playback_time():
    buffer = DelayedJointCommandBuffer(delay_s=0.08)
    buffer.reset(np.array([0.0, 2.0]), {"left": 0.0}, time_s=1.00)
    buffer.push(np.array([1.0, 4.0]), {"left": 1.0}, time_s=1.04)

    q, openings = buffer.sample(1.10)

    np.testing.assert_allclose(q, [0.5, 3.0])
    assert openings["left"] == pytest.approx(0.5)


def test_delayed_buffer_holds_latest_command_on_underflow():
    buffer = DelayedJointCommandBuffer(delay_s=0.08)
    buffer.reset(np.array([0.0]), {"left": 0.0}, time_s=1.00)
    buffer.push(np.array([1.0]), {"left": 1.0}, time_s=1.04)

    q, openings = buffer.sample(1.20)

    np.testing.assert_allclose(q, [1.0])
    assert openings == {"left": 1.0}


def test_reset_discards_an_old_trajectory_epoch():
    buffer = DelayedJointCommandBuffer(delay_s=0.08)
    buffer.reset(np.array([0.0]), {"left": 0.0}, time_s=1.00)
    buffer.push(np.array([10.0]), {"left": 1.0}, time_s=1.04)

    buffer.reset(np.array([3.0]), {"left": 0.25}, time_s=2.00)
    q, openings = buffer.sample(2.50)

    np.testing.assert_allclose(q, [3.0])
    assert openings == {"left": 0.25}


def test_player_publishes_from_its_fixed_rate_thread():
    published: list[float] = []
    received = threading.Event()

    def write(q, openings):
        del openings
        published.append(float(q[0]))
        if len(published) >= 3:
            received.set()

    player = DelayedJointCommandPlayer(
        write,
        command_rate_hz=100.0,
        delay_s=0.0,
    )
    player.start(np.array([2.0]), {"left": 0.5}, time_s=time.perf_counter())
    try:
        assert received.wait(timeout=0.2)
    finally:
        player.stop()

    assert published[:3] == [2.0, 2.0, 2.0]
    latest = player.latest()
    assert latest is not None
    np.testing.assert_allclose(latest[0], [2.0])
    assert latest[1] == {"left": 0.5}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"delay_s": -0.1}, "delay_s"),
        ({"delay_s": 0.1, "max_commands": 1}, "max_commands"),
    ),
)
def test_delayed_buffer_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        DelayedJointCommandBuffer(**kwargs)
