"""tashkeel: a small, open Arabic diacritization (tashkeel) model."""
from .model import Diacritizer
from .text import join_text, split_text, strip_diacritics

__all__ = ["Diacritizer", "join_text", "split_text", "strip_diacritics"]
__version__ = "0.1.0"
