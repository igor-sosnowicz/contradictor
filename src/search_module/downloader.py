"""Web page downloader implementation."""

from threading import local
from typing import override

import requests
from loguru import logger

from src.search_module.config import DownloadConfig
from src.search_module.interfaces import Downloader


class RequestsDownloader(Downloader):
    """
    Downloads web pages using requests.

    Each thread gets its own HTTP session.
    """

    def __init__(
        self,
        config: DownloadConfig,
    ) -> None:
        """Initialize downloader with HTTP configuration."""
        self.config = config
        self._thread_local = local()

    def _get_session(self) -> requests.Session:
        """Return the HTTP session belonging to the current thread."""
        session = getattr(self._thread_local, "session", None)

        if session is None:
            session = requests.Session()
            session.headers.update(
                {
                    "User-Agent": self.config.user_agent,
                }
            )
            self._thread_local.session = session

        return session

    @override
    def download(
        self,
        url: str,
    ) -> str:
        try:
            response = self._get_session().get(
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
