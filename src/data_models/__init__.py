"""
Data models package for the argument database system.

This package contains Pydantic models for shared domain concepts such as
interpretative frames. The argument database schema itself lives in
``src.contr_argument_dataset_tools.argument_schema``.
"""

from src.data_models.data_models import (
    FramedArgument,
    InterpretativeFrame,
    SubsetName,
    TextSpan,
)

__all__ = [
    "FramedArgument",
    "InterpretativeFrame",
    "SubsetName",
    "TextSpan",
]
