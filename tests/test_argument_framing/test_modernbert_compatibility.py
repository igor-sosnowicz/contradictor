"""Tests for ModernBERT training configuration compatibility."""

from pathlib import Path

import pytest
from huggingface_hub.errors import EntryNotFoundError

from src.argument_framing.argument_framing_dataset import ArgumentFramingDataset
from src.argument_framing.modernbert_framer.config import (
    RemoteTrainingMetadata,
    TrainingConfig,
    retrain_if_changed_differences,
)
from src.argument_framing.modernbert_framer.modernbert_framer import ModernBertFramer

EXPECTED_RETRAIN_IF_CHANGED_FIELDS = frozenset(
    {
        "max_training_epochs",
        "random_seed",
        "strategy",
        "learning_rate",
        "max_samples_per_split",
    }
)
PROOF_OF_CONCEPT_MAX_SAMPLES = 100


def test_retrain_if_changed_field_names_match_expected_fields() -> None:
    """Test that only fields affecting a trained model are marked."""
    assert (
        TrainingConfig.retrain_if_changed_field_names()
        == EXPECTED_RETRAIN_IF_CHANGED_FIELDS
    )


def test_retrain_if_changed_fields_exclude_runtime_only_fields() -> None:
    """Test that runtime-only fields are excluded from compatibility checks."""
    fields = TrainingConfig(
        batch_size=16,
        gradient_accumulation_steps=1,
        bfloat16=False,
    ).retrain_if_changed_fields()

    assert set(fields) == EXPECTED_RETRAIN_IF_CHANGED_FIELDS
    assert "batch_size" not in fields
    assert "gradient_accumulation_steps" not in fields
    assert "bfloat16" not in fields


def test_retrain_if_changed_fingerprint_ignores_runtime_only_fields() -> None:
    """Test that runtime-only fields do not change the compatibility hash."""
    first = TrainingConfig(batch_size=1, gradient_accumulation_steps=1, bfloat16=True)
    second = TrainingConfig(
        batch_size=32,
        gradient_accumulation_steps=16,
        bfloat16=False,
    )

    assert (
        first.retrain_if_changed_fingerprint()
        == second.retrain_if_changed_fingerprint()
    )


@pytest.mark.parametrize(
    "update",
    [
        {"max_training_epochs": 5},
        {"random_seed": 42},
        {"strategy": "head-only"},
        {"learning_rate": 3e-5},
        {"max_samples_per_split": 100},
    ],
)
def test_retrain_if_changed_fingerprint_changes_with_mattering_fields(
    update: dict[str, object],
) -> None:
    """Test that every retraining-relevant field changes the compatibility hash."""
    base = TrainingConfig()
    changed = TrainingConfig.model_validate(update)

    assert (
        base.retrain_if_changed_fingerprint()
        != changed.retrain_if_changed_fingerprint()
    )


def test_retrain_if_changed_differences_reports_only_changed_fields() -> None:
    """Test that differences contain exactly the incompatible fields."""
    requested = TrainingConfig(
        strategy="head-only",
        max_samples_per_split=100,
    )
    remote = TrainingConfig(
        strategy="extended-lora-and-head",
        max_samples_per_split=0,
    )

    differences = retrain_if_changed_differences(requested, remote)

    assert set(differences) == {"strategy", "max_samples_per_split"}
    assert differences["strategy"] == ("head-only", "extended-lora-and-head")
    assert differences["max_samples_per_split"] == (100, 0)


def test_remote_training_metadata_round_trip(tmp_path: Path) -> None:
    """Test that remote training metadata can be serialised and loaded again."""
    metadata = RemoteTrainingMetadata(
        training_config=TrainingConfig(max_samples_per_split=100),
        dataset_version="1.0.0",
    )
    metadata_path = tmp_path / "training_config.json"
    metadata_path.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")

    loaded = RemoteTrainingMetadata.model_validate_json(
        metadata_path.read_text(encoding="utf-8")
    )

    assert loaded == metadata


def test_proof_of_concept_mode_forces_max_samples_per_split() -> None:
    """Test that proof-of-concept mode records the subsampled training data."""
    framer = ModernBertFramer(
        dataset=ArgumentFramingDataset(),
        training_config=TrainingConfig(max_samples_per_split=0),
        proof_of_concept_mode=True,
    )

    assert framer._training_config.max_samples_per_split == PROOF_OF_CONCEPT_MAX_SAMPLES


def test_max_samples_per_split_changes_training_directory() -> None:
    """Test that full-data and proof-of-concept models use distinct directories."""
    full_framer = ModernBertFramer(
        dataset=ArgumentFramingDataset(),
        training_config=TrainingConfig(max_samples_per_split=0),
    )
    poc_framer = ModernBertFramer(
        dataset=ArgumentFramingDataset(),
        training_config=TrainingConfig(max_samples_per_split=100),
    )

    assert full_framer._training_directory != poc_framer._training_directory


def test_remote_training_config_matches_requested_config() -> None:
    """Test that a matching remote configuration is accepted."""
    training_config = TrainingConfig()
    framer = ModernBertFramer(
        dataset=ArgumentFramingDataset(),
        training_config=training_config,
    )
    metadata = RemoteTrainingMetadata(
        training_config=training_config,
        dataset_version=framer._dataset.version,
    )

    assert framer._is_remote_training_config_compatible(metadata)


def test_remote_training_config_rejects_different_dataset_version() -> None:
    """Test that a remote model trained on another dataset version is rejected."""
    training_config = TrainingConfig()
    framer = ModernBertFramer(
        dataset=ArgumentFramingDataset(),
        training_config=training_config,
    )
    metadata = RemoteTrainingMetadata(
        training_config=training_config,
        dataset_version="0.0.0",
    )

    assert not framer._is_remote_training_config_compatible(metadata)


def test_remote_model_without_training_metadata_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that a remote model without metadata falls back to training."""
    framer = ModernBertFramer(dataset=ArgumentFramingDataset())

    def missing_metadata(*_: object, **__: object) -> str:
        raise EntryNotFoundError("training_config.json is missing")

    def forbidden_model_load(*_: object, **__: object) -> object:
        raise AssertionError("Remote model weights must not be loaded.")

    monkeypatch.setattr(
        "src.argument_framing.modernbert_framer.modernbert_framer.hf_hub_download",
        missing_metadata,
    )
    monkeypatch.setattr(
        "src.argument_framing.modernbert_framer.modernbert_framer."
        "AutoTokenizer.from_pretrained",
        forbidden_model_load,
    )
    monkeypatch.setattr(
        "src.argument_framing.modernbert_framer.modernbert_framer."
        "ModernBertForSequenceClassification.from_pretrained",
        forbidden_model_load,
    )

    assert framer._load_from_hugging_face() == (None, None, None)


def test_remote_model_with_incompatible_training_config_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that incompatible remote metadata prevents loading the weights."""
    framer = ModernBertFramer(
        dataset=ArgumentFramingDataset(),
        training_config=TrainingConfig(strategy="head-only"),
    )
    metadata = RemoteTrainingMetadata(
        training_config=TrainingConfig(strategy="extended-lora-and-head"),
        dataset_version=framer._dataset.version,
    )
    metadata_path = tmp_path / "training_config.json"
    metadata_path.write_text(metadata.model_dump_json(), encoding="utf-8")

    def download_metadata(*_: object, **__: object) -> str:
        return str(metadata_path)

    def forbidden_model_load(*_: object, **__: object) -> object:
        raise AssertionError("Remote model weights must not be loaded.")

    monkeypatch.setattr(
        "src.argument_framing.modernbert_framer.modernbert_framer.hf_hub_download",
        download_metadata,
    )
    monkeypatch.setattr(
        "src.argument_framing.modernbert_framer.modernbert_framer."
        "AutoTokenizer.from_pretrained",
        forbidden_model_load,
    )
    monkeypatch.setattr(
        "src.argument_framing.modernbert_framer.modernbert_framer."
        "ModernBertForSequenceClassification.from_pretrained",
        forbidden_model_load,
    )

    assert framer._load_from_hugging_face() == (None, None, None)
