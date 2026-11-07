"""bahr: a small, open classifier for the meter (البحر) of classical Arabic verse."""

from .features import halves, rhythm_skeleton, strip_diacritics
from .labels import METERS, METERS_AR
from .model import MeterClassifier

__all__ = [
    "METERS",
    "METERS_AR",
    "MeterClassifier",
    "halves",
    "rhythm_skeleton",
    "strip_diacritics",
]
__version__ = "0.1.0"
