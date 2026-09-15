"""
Adapters normalizing raw argument datasets into the canonical schema.

Importing this package registers every per-source adapter, so callers only
need ``raw_schema.normalize(source_name, frame)``.
"""

from src.argument_dataset.datasets_handlers import adapters

__all__ = ["adapters"]
