"""calibcheck: measure and fix the calibration of classifier probabilities."""
from .metrics import (
    accuracy,
    brier_decomposition,
    brier_score,
    calibration_report,
    expected_calibration_error,
    log_loss,
    maximum_calibration_error,
    reliability_bins,
)
from .calibrators import CALIBRATORS, IsotonicCalibrator, PlattScaling, TemperatureScaling

__version__ = "0.1.0"
__all__ = [
    "accuracy", "brier_decomposition", "brier_score", "calibration_report",
    "expected_calibration_error", "log_loss", "maximum_calibration_error",
    "reliability_bins", "CALIBRATORS", "IsotonicCalibrator", "PlattScaling",
    "TemperatureScaling",
]
