"""Common interfaces shared across application modules."""

from abc import ABC, abstractmethod

from src.data_models.abstract.configuration import ConfigurationValue


class ConfigurationProvider(ABC):
    """Base interface for components exposing their configuration."""

    @abstractmethod
    def get_configuration(self) -> dict[str, ConfigurationValue]:
        """
        Get the configuration of the component.

        Returns:
            dict[str, ConfigurationValue]: A dictionary containing the
                component's configuration parameters and their values.
        """
