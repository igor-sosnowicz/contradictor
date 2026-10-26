"""Module for searching in a local dataset."""

from typing import override

from src.benchmark.golden_dataset import GoldenDataset
from src.search_module.interfaces import SearchEngine
from src.search_module.models import SearchQuery, SearchResult


class LocalDatasetSearchEngine(SearchEngine):
    """Provider of all samples from a local dataset."""

    def __init__(self, dataset: GoldenDataset) -> None:
        """
        Initialise a search engine of the Golden Dataset raw texts.

        Args:
            dataset (GoldenDataset): A dataset to search texts in.
        """
        self._dataset = dataset

    @override
    def search(
        self,
        query: SearchQuery,
        limit: int = 10,
    ) -> list[SearchResult]:
        return [
            SearchResult(
                url=str(path),
                title=text[:30],
                snippet=text,
            )
            for path, text in self._dataset.raw_texts.items()
        ]
