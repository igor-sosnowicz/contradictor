"""Paths owned by the argument framing module."""

from enum import Enum

from src.paths.core import CorePaths

_MODELS = CorePaths.MODELS_DIR.value


class ArgumentFramingPaths(Enum):
    """Wishlist of the argument framing module."""

    FRAME_CLASSIFIER_FILE = (_MODELS / "xgboost_frame_classifier.pkl").as_file()
    TFIDF_VECTORIZER_FILE = (_MODELS / "tfidf_vectorizer.pkl").as_file()
