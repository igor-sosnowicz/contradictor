"""Web page downloader implementation."""

from typing import override

import requests
from loguru import logger

from src.search_module.config import DownloadConfig
from src.search_module.interfaces import Downloader


class RequestsDownloader(Downloader):
    """
    Downloads web pages using requests.

    Responsible only for HTTP communication.
    """

    def __init__(
        self,
        config: DownloadConfig,
    ) -> None:
        """Initialize web page downloader with HTTP configuration."""
        self.config = config
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": self.config.user_agent,
            }
        )

    @override
    def download(
        self,
        url: str,
    ) -> str:
        try:
            response = self.session.get(
                url,
                timeout=self.config.timeout,
                allow_redirects=True,
            )
            response.raise_for_status()
        except requests.RequestException:
            logger.exception(
                "Failed downloading URL: {}",
                url,
            )
            return ""

        return response.text
