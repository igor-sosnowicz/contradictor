"""Module for providing files from a local source."""

from pathlib import Path
from typing import override

from src.benchmark.golden_dataset import GoldenDataset
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
    def download(
        self,
        url: str,
    ) -> str:
        return self._dataset.raw_texts[Path(url)]
