"""Automatic room segmentation of one merged, z-up, single-storey cloud (spec 2026-10-01)."""
from .core import segment
from .errors import SegmentationError
from .params import DEFAULTS, HOVSG, Params

__all__ = ["segment", "SegmentationError", "Params", "DEFAULTS", "HOVSG"]
