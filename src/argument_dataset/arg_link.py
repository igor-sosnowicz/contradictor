"""Link types between arguments and their counter-arguments."""

from enum import StrEnum, auto


class ArgumentLinkType(StrEnum):
    """Enumeration for argument link types (see docs/dataset_building_pipeline.md)."""

    CONTRARGUMENT = auto()
    COUNTEREXAMPLE = auto()
    CONTRADICTION = auto()
    REBUTTAL = auto()
