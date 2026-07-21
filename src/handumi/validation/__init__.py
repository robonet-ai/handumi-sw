"""Software groundwork for TEST-001; laboratory validation is still required."""

from handumi.validation.core import (
    DropoutInterval,
    FrameTransform,
    SyncEvent,
    bootstrap_participant_mean,
    classification_metrics,
    dropout_intervals,
    orientation_error_deg,
    position_errors,
)

__all__ = [
    "DropoutInterval",
    "FrameTransform",
    "SyncEvent",
    "bootstrap_participant_mean",
    "classification_metrics",
    "dropout_intervals",
    "orientation_error_deg",
    "position_errors",
]
