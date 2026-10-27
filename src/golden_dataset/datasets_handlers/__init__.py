"""
Initialization file for the datasets_handlers package.

'adapters.py' are loaded eagerly to ensure that all adapters are registered
within the registry.
"""

from src.golden_dataset.datasets_handlers import adapters

__all__ = ["adapters"]
