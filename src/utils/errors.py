"""Module with hierarchy of exceptions."""

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)


class ContradictorError(Exception):
    """Root-level error from which all Contradictor errors inherit."""


class DatasetError(ContradictorError):
    """General issue with a dataset."""


class ModelNotTrainedError(ContradictorError):
    """Raised when a trained model or vectorizer is missing."""


class NotPreparedError(ContradictorError):
    """Raised if an entity is not prepared but was attempted to run."""


class ConfigurationError(ContradictorError):
    """Raised if an invalid value was set in the configuration."""


class UnsupportedError(ContradictorError):
    """Raised if a currently unsupported option or combination of options is used."""


class RetryExhaustedError(ContradictorError):
    """Raised when all retry attempts for an operation fail."""


RETRIABLE_LM_STUDIO_ERRORS: tuple[type[BaseException], ...] = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
)
