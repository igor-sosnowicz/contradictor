"""Module for providing files from a local source."""

from pathlib import Path
from types import TracebackType
from typing import Self, override

from src.benchmark.golden_dataset import GoldenDataset
from src.data_models.abstract.configuration import ConfigurationValue
from src.search_module.interfaces import Downloader


class LocalDownloader(Downloader):
    """Downloader for local samples."""

    def __init__(self, dataset: GoldenDataset) -> None:
        """
        Initialise a local downloader.

        Args:
            dataset (GoldenDataset): Dataset to find raw text samples in.
        """
        super().__init__()
        self._dataset = dataset

    @override
    async def download(
        self,
        url: str,
    ) -> str:
        return self._dataset.raw_texts[Path(url)]

    @override
    def get_configuration(self) -> dict[str, ConfigurationValue]:
        return {
            "implementation": self.__class__.__name__,
        }

    @override
    async def __aenter__(self) -> Self:
        return self

    @override
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> bool | None:
        """Close the asynchronous HTTP client."""
        return None
