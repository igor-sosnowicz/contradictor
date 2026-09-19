"""Module with hierarchy of exceptions."""


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


class NLIResultError(ContradictorError):
    """Raised if an arbitrary instead of a valid value was used for NLIResult."""
