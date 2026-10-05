"""Paths owned by the argument detection module."""

from enum import Enum

from src.paths.core import CorePaths

_PROCESSED = CorePaths.PROCESSED_DATASETS_DIR.value


class ArgumentDetectionPaths(Enum):
    """Wishlist of the argument detection module."""

    CLAIM_PROCESSED_DIR = _PROCESSED / "claim_extraction"
    EVIDENCE_PROCESSED_DIR = _PROCESSED / "evidence_extraction"
