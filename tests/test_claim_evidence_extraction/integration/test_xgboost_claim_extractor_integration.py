"""Integration tests for the claim extractior."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from src.argument_detection.argument_detection_dataset import ArgumentDetectionDataset
from src.argument_detection.config import ThresholdConfig
from src.argument_detection.xgboost_claim_extractor import XGBoostClaimExtractor
from src.configuration import config
from src.data_models.data_models import SubsetName


class MiniIntegrationDataset(ArgumentDetectionDataset):
    """
    Provides a small, deterministic dataset for integration tests.

    The dataset is fully local and does not require access to Kaggle or any
    external network resources.
    """

    def __init__(self) -> None:
        """Initialize the deterministic integration-test dataset."""

    async def prepare(self) -> None:
        """Mark the dataset as prepared."""
        self._is_prepared = True

    def get_claim_split(
        self,
        subset: SubsetName,
        max_samples: int | None = None,
    ) -> pd.DataFrame:
        """
        Return a small deterministic claim classification dataset.

        Args:
            subset (SubsetName): Dataset subset requested by the caller.
            max_samples (int | None): Maximum number of samples to return.

        Returns:
            pd.DataFrame: DataFrame containing sentences and their claim
                labels.
        """
        data = pd.DataFrame(
            {
                "sentence": [
                    "Cats are intelligent creatures.",
                    "The Earth revolves around the Sun.",
                    "We should reduce carbon emissions.",
                    "I went to the store yesterday.",
                    "The sky is blue.",
                    "Please close the door.",
                ],
                "is_claim": [1, 1, 1, 0, 0, 0],
            }
        )
        return data.head(max_samples) if max_samples is not None else data


@pytest.fixture
def isolated_ml_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """
    Create an isolated filesystem environment with a test XGBoost model.

    The model is trained using the real embedder and XGBoost classifier. All
    model artifacts are written to pytest's temporary directory.

    Args:
        tmp_path (Path): Pytest-provided temporary directory.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture used to override
            application configuration.

    Returns:
        Path: Temporary directory containing the test model.
    """
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True)

    monkeypatch.setattr(config, "data_directory", tmp_path)
    monkeypatch.setattr(config, "model_subdirectory", Path("models"))

    dataset = MiniIntegrationDataset()
    extractor = XGBoostClaimExtractor(dataset, proof_of_concept_mode=True)

    training_sentences = [
        "Cats are intelligent creatures.",
        "The Earth revolves around the Sun.",
        "We should reduce carbon emissions.",
        "I went to the store yesterday.",
        "The sky is blue.",
        "Please close the door.",
    ]
    X_train = extractor._embedder.embed(training_sentences)
    y_train = np.array([1, 1, 1, 0, 0, 0])

    mini_xgb = XGBClassifier(
        n_estimators=10,
        max_depth=2,
        learning_rate=0.3,
        random_state=42,
        eval_metric="logloss",
    )
    mini_xgb.fit(X_train, y_train)

    model_path = model_dir / "xgboost_claim_extractor.json"
    mini_xgb.save_model(str(model_path))

    assert model_path.exists()
    assert model_path.stat().st_size > 0

    return tmp_path


@pytest.mark.asyncio
async def test_claim_extractor_e2e_flow(isolated_ml_env: Path) -> None:
    """
    Verify the complete claim extraction flow without mocks.

    The test uses the real embedder, XGBoost model, configuration, and
    filesystem while keeping all model artifacts isolated in a temporary
    directory.

    Args:
        isolated_ml_env (Path): Temporary directory containing the test model.
    """
    dataset = MiniIntegrationDataset()
    extractor = XGBoostClaimExtractor(dataset, proof_of_concept_mode=True)

    low_threshold_config = ThresholdConfig(claim=0.01)
    extractor._config = extractor._config.model_copy(
        update={"threshold": low_threshold_config}
    )

    text_to_test = (
        "Cats are intelligent creatures. Testing integration systems right now."
    )
    extracted_claims = await extractor.extract_claims(text_to_test)

    assert isinstance(extracted_claims, list)
    assert extracted_claims, (
        "Extractor did not return any claims. The integration pipeline may be broken."
    )
    assert all(isinstance(claim, str) and claim.strip() for claim in extracted_claims)
    assert any(
        "Cats are intelligent creatures" in claim for claim in extracted_claims
    ), f"Expected the claim sentence to be extracted. Actual claims: {extracted_claims}"


@pytest.mark.asyncio
async def test_claim_extractor_training_flow(isolated_ml_env: Path) -> None:
    """
    Verify model training and persistence without mocks.

    Args:
        isolated_ml_env (Path): Temporary directory where the model is
            expected to be saved.
    """
    dataset = MiniIntegrationDataset()
    extractor = XGBoostClaimExtractor(dataset, proof_of_concept_mode=True)

    trained_model = await extractor._train_model()

    expected_model_path = isolated_ml_env / "models" / "xgboost_claim_extractor.json"

    assert isinstance(trained_model, XGBClassifier)
    assert expected_model_path.exists()
    assert expected_model_path.is_file()
    assert expected_model_path.stat().st_size > 0
