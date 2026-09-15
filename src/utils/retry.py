"""
Retry decorator with exponential backoff for transient failures.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import ParamSpec, Protocol, TypeVar

from src.utils.errors import RetryExhaustedError

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


class RetryPolicy(Protocol):
    """Structural retry policy consumed by the retry decorator."""

    @property
    def max_retries(self) -> int:
        """Maximum retry attempts after the first call."""
        ...

    @property
    def base_delay_seconds(self) -> float:
        """Delay before the first retry; doubles on every attempt."""
        ...

    @property
    def max_delay_seconds(self) -> float:
        """Upper bound for the delay between retries."""
        ...


@dataclass(frozen=True)
class RetrySettings:
    """Per-call-site retry policy for the retry decorator."""

    max_retries: int = 4
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0

    def __post_init__(self) -> None:
        """Validate policy values."""
        if (
            self.max_retries < 0
            or self.base_delay_seconds <= 0
            or self.max_delay_seconds <= 0
        ):
            raise ValueError(
                "RetrySettings needs max_retries >= 0 and positive delays."
            )


DEFAULT_RETRY_POLICY = RetrySettings()


def _resolve_policy(
    policy: RetryPolicy | None, first_arg: object | None
) -> RetryPolicy:
    if policy is not None:
        return policy

    candidate = getattr(first_arg, "retry", None)

    if (
        candidate is not None
        and hasattr(candidate, "max_retries")
        and hasattr(candidate, "base_delay_seconds")
        and hasattr(candidate, "max_delay_seconds")
    ):
        return candidate

    return DEFAULT_RETRY_POLICY


def retry(
    policy: RetryPolicy | None = None,
    *,
    on: tuple[type[BaseException], ...],
    operation: str | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Retry the wrapped callable with exponential backoff.
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        label = operation or func.__qualname__

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            active = _resolve_policy(policy, args[0] if args else None)
            total_attempts = active.max_retries + 1

            for attempt in range(total_attempts):
                try:
                    return func(*args, **kwargs)
                except on as e:
                    if attempt >= active.max_retries:
                        raise RetryExhaustedError(
                            f"{label} failed after {total_attempts} attempts"
                        ) from e

                    delay = min(
                        active.base_delay_seconds * (2**attempt),
                        active.max_delay_seconds,
                    )

                    logger.warning(
                        "%s failed (attempt %d/%d): %s. Retrying in %.1fs.",
                        label,
                        attempt + 1,
                        total_attempts,
                        e,
                        delay,
                    )

                    time.sleep(delay)

            raise RetryExhaustedError(f"{label} did not run")

        return wrapper

    return decorator
