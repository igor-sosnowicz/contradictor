"""Module with tests for XGBoost-based argument detector."""

import pytest

from src.argument_detection.argument_detection_dataset import ArgumentDetectionDataset
from src.argument_detection.xgboost_detector import XGBoostDetector


@pytest.fixture
def argument_detection_dataset() -> ArgumentDetectionDataset:
    """Fixture providing an argument detection dataset."""
    return ArgumentDetectionDataset()


@pytest.mark.asyncio
async def test_end_to_end_detection(
    argument_detection_dataset: ArgumentDetectionDataset,
) -> None:
    """Test end-to-end argument detection with XGBoost."""
    detector = XGBoostDetector(
        argument_detection_dataset,
        proof_of_concept_mode=True,
    )

    text = """
    Testing software thoroughly reduces the chance of hidden bugs reaching users.
    It protects the company's reputation. It lowers the cost of fixing defects early.
    It improves security by exposing vulnerabilities before release.
    It supports smoother maintenance by making changes safer.
    It increases user trust through more reliable behavior.
    It helps teams catch integration issues between components.
    It reduces downtime and support burden after launch.
    It makes performance problems visible before they become critical.
    It ultimately leads to a better product that works as intended in the real-world."""
    arguments = await detector.detect(text=text)
    assert isinstance(arguments, list)
    if arguments:
        for argument in arguments:
            assert isinstance(argument, str)
