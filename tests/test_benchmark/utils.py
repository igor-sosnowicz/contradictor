"""Module with benchmark testing utility functions."""

import json
from datetime import datetime
from pathlib import Path
from typing import Final

from src.benchmark.command import BENCHMARK_RESULTS_DIR


def _to_second_since_unix(filename: str) -> int:
    """
    Convert a benchmark result filename to seconds since the Unix epoch.

    Args:
        filename (str): Name of a benchmark result file. Its stem must match the
            format `%H_%M_%S_%d_%m_%Y_<uuid>`, where `<uuid>` is a UUID string
            appended to the timestamp to make filenames unique across runs.

    Returns:
        int: Number of seconds since the Unix epoch for the parsed timestamp.

    Raises:
        ValueError: If the filename stem does not match the expected
            `%H_%M_%S_%d_%m_%Y_<uuid>` format or the timestamp part cannot
            be parsed.
    """
    # Number of leading parts that form the timestamp: hour, minute, second,
    # day, month, year.
    timestamp_part_count: Final[int] = 6
    # Expected total underscore-separated parts in a benchmark result filename
    # stem: the timestamp parts plus the uuid.
    expected_filename_parts: Final[int] = timestamp_part_count + 1

    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) < expected_filename_parts:
        raise ValueError(
            f"Benchmark result filename '{filename}' does not match the expected "
            f"format '%H_%M_%S_%d_%m_%Y_<uuid>'. Got stem '{stem}' with "
            f"{len(parts)} underscore-separated parts (expected at least "
            f"{expected_filename_parts}: "
            f"hour, minute, second, day, month, year, uuid)."
        )
    timestamp_part = "_".join(parts[:timestamp_part_count])
    try:
        dt = datetime.strptime(
            timestamp_part,
            "%H_%M_%S_%d_%m_%Y",
        ).astimezone()
    except ValueError as exc:
        raise ValueError(
            f"Benchmark result filename '{filename}' has a timestamp part "
            f"'{timestamp_part}' that does not match '%H_%M_%S_%d_%m_%Y'."
        ) from exc

    return int(dt.timestamp())


def validate_result(data: dict) -> None:
    """
    Validate that a benchmark result dictionary contains the expected structure.

    Args:
        data (dict): Benchmark result dictionary loaded from a result file.

    Raises:
        ValueError: Raised if the result dictionary is empty.
        AssertionError: Raised if `metrics`, `configuration`, or `metadata` keys
            are missing or not dictionaries, or if any required metadata field
            is missing or has an invalid type.
    """
    if not data:
        raise ValueError("The benchmark result is empty.")
    expected_keys = ["metrics", "configuration", "metadata"]
    for expected_key in expected_keys:
        assert expected_key in data
        assert isinstance(data[expected_key], dict)

    metadata = data["metadata"]
    required_metadata_fields: list[tuple[str, tuple[type, ...]]] = [
        ("start_datetime", (str,)),
        ("finish_datetime", (str,)),
        ("duration", (int, float)),
        ("golden_dataset_version", (str,)),
    ]
    for field, expected_types in required_metadata_fields:
        assert field in metadata, f"Missing metadata field: {field}"
        value = metadata[field]
        assert isinstance(value, expected_types), (
            f"Metadata field '{field}' must be {expected_types}, got {type(value)}"
        )
    assert metadata["duration"] >= 0, "Metadata 'duration' must be non-negative."
    assert metadata["start_datetime"], "Metadata 'start_datetime' must not be empty."
    assert metadata["finish_datetime"], "Metadata 'finish_datetime' must not be empty."
    assert metadata["golden_dataset_version"], (
        "Metadata 'golden_dataset_version' must not be empty."
    )


def load_latest_benchmark_result(
    results_directory: Path = BENCHMARK_RESULTS_DIR,
) -> dict:
    """
    Load the most recent benchmark result file from the results directory.

    Args:
        results_directory (Path): Directory containing benchmark result files.
            Defaults to `BENCHMARK_RESULTS_DIR`.

    Returns:
        dict: Parsed JSON content of the latest benchmark result file.
    """
    assert results_directory.is_dir()

    latest_timestamp = 0
    current_directory: Final[Path] = Path()
    latest_file = current_directory
    for file in results_directory.iterdir():
        seconds = _to_second_since_unix(file.name)
        if seconds > latest_timestamp:
            latest_timestamp = seconds
            latest_file = file

    assert latest_file != current_directory, "There are no result files."

    with latest_file.open("rt") as file:
        return json.load(file)
