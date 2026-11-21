"""Web page downloader implementation."""

from types import TracebackType
from typing import Self, override

import httpx
from httpx import AsyncClient
from loguru import logger

from src.data_models.abstract.configuration import ConfigurationValue
from src.search_module.config import DownloadConfig
from src.search_module.interfaces import Downloader
from src.utils.errors import NotPreparedError


class HttpxDownloader(Downloader):
    """
    Downloads web pages using httpx.

    Uses a shared asynchronous HTTP client.
    """

    def __init__(
        self,
        config: DownloadConfig,
    ) -> None:
        """
        Initialise downloader with HTTP configuration.

        Args:
            config (DownloadConfig): Configuration containing HTTP settings,
                such as timeout and user agent.
        """
        self.config = config
        self._client: AsyncClient | None = None

    @override
    async def __aenter__(self) -> Self:
        self._client = httpx.AsyncClient(
            headers={
                "User-Agent": self.config.user_agent,
            },
            timeout=self.config.timeout,
            follow_redirects=True,
            limits=httpx.Limits(
                max_connections=self.config.max_connections,
                max_keepalive_connections=self.config.max_keepalive_connections,
                keepalive_expiry=self.config.keepalive_expiry,
            ),
        )

        return self

    @override
    async def download(
        self,
        url: str,
    ) -> str:
        if self._client is None:
            raise NotPreparedError(
                f"{HttpxDownloader.__name__}.{self.download.__name__}(...) can only be "
                "used inside an async context manager. Try: async with "
                f"{HttpxDownloader.__name__} as downloader:\n\tawait "
                f"downloader.{self.download.__name__}(...)"
            )
        try:
            response = await self._client.get(url)
            response.raise_for_status()

        except httpx.HTTPError:
            logger.exception(
                "Failed downloading URL: {}",
                url,
            )
            return ""

        return response.text

    @override
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> bool | None:
        """Close the asynchronous HTTP client."""
        if self._client is not None:
            await self._client.aclose()
        return False

    @override
    def get_configuration(self) -> dict[str, ConfigurationValue]:
        return {
            "implementation": self.__class__.__name__,
            "timeout": self.config.timeout,
            "user_agent": self.config.user_agent,
        }
