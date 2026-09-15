"""
Progress logging for the slow, LLM-bound loops.

Every step here costs an LLM round trip, so progress is logged per item and
before the work starts - you see what is being worked on while it happens,
not a silent terminal for ten minutes:

    Action |   3/20 | 12s elapsed, ~68s left
    Action |   4/20 | 16s elapsed, ~64s left
    Action | done 20 in 1m20s
"""

import time
from collections.abc import Generator, Sequence

SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600


def format_duration(seconds: float) -> str:
    """Render a duration as 45s / 2m30s / 1h04m."""
    if seconds < SECONDS_PER_MINUTE:
        return f"{seconds:.0f}s"
    if seconds < SECONDS_PER_HOUR:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h{int((seconds % 3600) // 60):02d}m"


def track[T](items: Sequence[T], task: str) -> Generator[T]:
    """
    Yield items, logging "task | i/N | elapsed, eta" before each one.

    Logs a closing line with the total duration so a finished stage is
    obvious in the terminal.
    """
    from loguru import logger

    total = len(items)
    if not total:
        logger.info(f"{task} | nothing to do")
        return

    started = time.monotonic()
    for index, item in enumerate(items, start=1):
        elapsed = time.monotonic() - started
        eta = ""
        if index > 1:
            remaining = (elapsed / (index - 1)) * (total - index + 1)
            eta = f", ~{format_duration(remaining)} left"
        logger.info(
            f"{task} | {index:>3}/{total} | {format_duration(elapsed)} elapsed{eta}"
        )
        yield item

    logger.info(
        f"{task} | done {total} in {format_duration(time.monotonic() - started)}"
    )
