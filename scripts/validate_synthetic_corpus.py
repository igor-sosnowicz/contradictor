"""
Validate a synthetic carrier batch and score extraction quality per slice.

Run with::

    uv run python scripts/validate_synthetic_corpus.py --batch-id 20260914_120000
"""

import argparse
import sys
from pathlib import Path

from loguru import logger

from src.argument_dataset.config import PipelineConfig
from src.argument_dataset.validation import ValidationReport, validate_batch


def main() -> int:
    """Validate one batch, returning a non-zero exit code on integrity issues."""
    parser = argparse.ArgumentParser(
        description="Validate a synthetic batch against the gold argument pool."
    )
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--batch-dir", type=Path, default=None)
    parser.add_argument("--gold-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = PipelineConfig()
    batch_dir: Path = args.batch_dir or cfg.synthetic_batch_dir(args.batch_id)
    gold_dir: Path = args.gold_dir or cfg.merged_gold_dir()

    report: ValidationReport = validate_batch(
        batch_dir,
        gold_dir,
        batch_id=args.batch_id,
        premise_threshold=cfg.premise_support_threshold,
        extracted_dir=cfg.extracted_synthetic_db_dir(args.batch_id),
    )
    logger.info(f"\n{report.render()}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
