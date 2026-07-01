"""Module with hierarchy of exceptions."""


class ContradictorError(Exception):
    """Root-level error from which all Contradictor errors inherit."""


class DatasetError(ContradictorError):
    """General issue with a dataset."""
