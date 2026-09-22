"""
Performance benchmark script for measuring
ReadabilityCleaner execution time and RAM usage.
"""

import json
import time
import tracemalloc
from pathlib import Path

from loguru import logger

from src.search_module.cleaner import ReadabilityCleaner
from src.search_module.config import CleaningConfig

BASE_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = BASE_DIR / "fixtures"

RESULTS_DIR = BASE_DIR.parents[1] / "benchmark-results-readability"
RESULTS_FILE = RESULTS_DIR / "experiment_results_readability.json"


def run_benchmark() -> None:
    """
    Execute the HTML cleaner performance benchmark and save aggregated results.

    Iterates through all discovered HTML files in the fixtures directory,
    profiles the execution time and peak memory consumption of the
    ReadabilityCleaner, and persists the structured metrics to a benchmark
    report file on disk.
    """
    html_files = sorted(FIXTURES_DIR.glob("*.html"))
    if not html_files:
        logger.info(
            f"Error: No HTML files found in directory: {FIXTURES_DIR.resolve()}"
        )
        return

    config = CleaningConfig(min_text_length=200)
    readability_cleaner = ReadabilityCleaner(config=config)

    results = {}

    logger.info(f"Starting performance analysis for {len(html_files)} files...")

    for html_path in html_files:
        html_content = html_path.read_text(encoding="utf-8")
        file_id = html_path.name
        url_stub = html_path.stem

        tracemalloc.start()
        start_time = time.perf_counter()

        try:
            readability_doc = readability_cleaner.clean(html_content, url=url_stub)
            readability_text = readability_doc.text or ""
            readability_title = readability_doc.title or "No Title"
        except (AttributeError, ValueError) as e:
            readability_text = ""
            readability_title = f"ERROR: {e!s}"

        readability_time = time.perf_counter() - start_time
        _, readability_peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        results[file_id] = {
            "readability": {
                "title": readability_title,
                "characters": len(readability_text),
                "words": len(readability_text.split()),
                "time_ms": readability_time * 1000,
                "peak_mem_kb": readability_peak_mem / 1024,
                "full_text": readability_text,
            }
        }

    logger.info("\n=== AGGREGATED BENCHMARK RESULTS ===")
    total_files = len(html_files)
    if total_files == 0:
        return

    avg_time_read = (
        sum(r["readability"]["time_ms"] for r in results.values()) / total_files
    )
    avg_mem_read = (
        sum(r["readability"]["peak_mem_kb"] for r in results.values()) / total_files
    )
    logger.info("Average processing time per page:")
    logger.info(f"  - Readability Cleaner: {avg_time_read:.2f} ms")
    logger.info("Average peak memory consumption:")
    logger.info(f"  - Readability Cleaner: {avg_mem_read:.2f} KB")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(
        f"\nFull benchmark report successfully saved to: {RESULTS_FILE.resolve()}"
    )


if __name__ == "__main__":
    run_benchmark()
