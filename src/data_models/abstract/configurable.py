"""Common interfaces shared across application modules."""

from abc import ABC, abstractmethod

from src.data_models.abstract.configuration import ConfigurationValue


class Configurable(ABC):
    """Base interface for components exposing their configuration."""

    @abstractmethod
    def get_configuration(self) -> dict[str, ConfigurationValue]:
        """Return the component configuration."""
