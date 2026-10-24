"""Module with a timer."""

import time
from types import TracebackType
from typing import Self

from loguru import logger


class Timer:
    """
    Synchronous context manager for measuring & logging execution time.

    The elapsed time is logged with a "[PERF]" prefix, so that performance logs
    can be filtered out from the rest of the application logs.

    Usage:
        with Timer("Dataset load"):
            dataset = load_dataset(...)
    """

    def __init__(self, task_name: str, log_level: int | str = "DEBUG") -> None:
        """
        Initialise the timer.

        Args:
            task_name (str): Human-readable name of the measured task.
            log_level (int | str): Level to log the result at. Accepts a level
                name ("DEBUG", "INFO", ...) or a numeric level (e.g. logging.DEBUG).
                Defaults to "DEBUG".
        """
        self.task_name = task_name
        self.log_level = log_level
        self.elapsed: float | None = None
        self._start: float = 0.0

    def __enter__(self) -> Self:
        """Start measuring and return the timer itself."""
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Stop measuring and log the elapsed time. Exceptions propagate unchanged."""
        self.elapsed = time.perf_counter() - self._start
        logger.log(
            self.log_level,
            f"[PERF] Task '{self.task_name}' took {self.elapsed:.3f}s",
        )


class AsyncTimer:
    """
    Asynchronous context manager for measuring & logging execution time.

    The elapsed time is logged with a "[PERF]" prefix, so that performance logs
    can be filtered out from the rest of the application logs.

    Usage:
        async with AsyncTimer("Encoder run"):
            await encoder.encode(texts)
    """

    def __init__(self, task_name: str, log_level: int | str = "DEBUG") -> None:
        """
        Initialise the timer.

        Args:
            task_name (str): Human-readable name of the measured task.
            log_level (int | str): Level to log the result at. Accepts a level
                name ("DEBUG", "INFO", ...) or a numeric level (e.g. logging.DEBUG).
                Defaults to "DEBUG".
        """
        self.task_name = task_name
        self.log_level = log_level
        self.elapsed: float | None = None
        self._start: float = 0.0

    async def __aenter__(self) -> Self:
        """Start measuring and return the timer itself."""
        self._start = time.perf_counter()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Stop measuring and log the elapsed time. Exceptions propagate unchanged."""
        self.elapsed = time.perf_counter() - self._start
        logger.log(
            self.log_level,
            f"[PERF] Task '{self.task_name}' took {self.elapsed:.3f}s",
        )
