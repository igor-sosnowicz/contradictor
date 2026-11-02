"""Web page downloader implementation."""

from typing import override

import httpx
from loguru import logger

from src.search_module.config import DownloadConfig
from src.search_module.interfaces import Downloader


class HttpxDownloader(Downloader):
    """
    Downloads web pages using httpx.

    Uses a shared asynchronous HTTP client.
    """

    def __init__(
        self,
        config: DownloadConfig,
    ) -> None:
        """Initialise downloader with HTTP configuration."""
        self.config = config

        self._client = httpx.AsyncClient(
            headers={
                "User-Agent": self.config.user_agent,
            },
            timeout=self.config.timeout,
            follow_redirects=True,
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
                keepalive_expiry=30.0,
            ),
        )

    @override
    async def download(
        self,
        url: str,
    ) -> str:
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

    async def close(self) -> None:
        """Close the asynchronous HTTP client."""
        await self._client.aclose()
