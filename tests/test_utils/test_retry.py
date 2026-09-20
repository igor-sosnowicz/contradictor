"""Module for testing the retry decorator."""

from unittest.mock import MagicMock

import pytest

from src.utils.errors import RetryExhaustedError
from src.utils.retry import RetrySettings, retry


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record backoff delays instead of sleeping."""
    delays: list[float] = []
    monkeypatch.setattr("src.utils.retry.time.sleep", delays.append)
    return delays


def test_retry_succeeds_after_transient_failures(no_sleep: list[float]) -> None:
    """Transient failures are retried with growing delays until success."""
    policy = RetrySettings(
        max_retries=3, base_delay_seconds=1.0, max_delay_seconds=30.0
    )
    flaky = MagicMock(side_effect=[ValueError("boom"), ValueError("boom"), "ok"])

    @retry(policy, on=(ValueError,), operation="flaky call")
    def call() -> str:
        return flaky()

    assert call() == "ok"
    assert flaky.call_count == 3
    assert no_sleep == [1.0, 2.0]


def test_retry_raises_domain_error_after_exhaustion(no_sleep: list[float]) -> None:
    """Exhaustion raises RetryExhaustedError chained from the last error."""
    policy = RetrySettings(
        max_retries=2, base_delay_seconds=1.0, max_delay_seconds=30.0
    )
    failing = MagicMock(side_effect=ValueError("down"))

    @retry(policy, on=(ValueError,), operation="failing call")
    def call() -> str:
        return failing()

    with pytest.raises(RetryExhaustedError, match="failing call failed after 3") as exc:
        call()

    assert isinstance(exc.value.__cause__, ValueError)
    assert failing.call_count == 3
    assert no_sleep == [1.0, 2.0]


def test_retry_uses_qualname_and_instance_policy(no_sleep: list[float]) -> None:
    """Methods default to their qualified name and pick up instance policy."""
    flaky = MagicMock(side_effect=[ValueError("boom"), "ok"])

    class Client:
        def __init__(self) -> None:
            self.retry = RetrySettings(
                max_retries=5, base_delay_seconds=1.0, max_delay_seconds=30.0
            )

        @retry(on=(ValueError,))
        def call(self) -> str:
            return flaky()

    assert Client().call() == "ok"
    assert flaky.call_count == 2
    assert no_sleep == [1.0]


def test_retry_ignores_unlisted_errors(no_sleep: list[float]) -> None:
    """Unlisted errors pass through without a retry."""
    innov = MagicMock(side_effect=TypeError("unexpected"))

    @retry(on=(ValueError,), operation="strict call")
    def call() -> str:
        return innov()

    with pytest.raises(TypeError, match="unexpected"):
        call()

    assert innov.call_count == 1
    assert no_sleep == []


def test_retry_settings_reject_invalid_values() -> None:
    """Negative retries and non-positive delays fail fast at construction."""
    with pytest.raises(ValueError, match="max_retries"):
        RetrySettings(max_retries=-1)

    with pytest.raises(ValueError, match="max_retries"):
        RetrySettings(base_delay_seconds=0.0)
