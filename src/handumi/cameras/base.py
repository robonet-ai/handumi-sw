"""Backend-neutral camera contracts for HandUMI recording."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraSample:
    """One camera frame timestamped on the workstation monotonic clock."""

    image: np.ndarray
    capture_time_ns: int
    sequence: int


class CameraDevice(ABC):
    """Minimal camera interface used by HandUMI recorders."""

    @abstractmethod
    def connect(self) -> None:
        """Open the camera stream."""

    @abstractmethod
    def async_read(self) -> np.ndarray:
        """Return the latest RGB frame."""

    @abstractmethod
    def sample_at(self, target_time_ns: int | None = None) -> CameraSample:
        """Return the buffered frame nearest ``target_time_ns``."""

    @abstractmethod
    def disconnect(self) -> None:
        """Close the camera stream."""
