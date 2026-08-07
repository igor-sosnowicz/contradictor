"""Configuration for XGBoost claim extractor."""

from pydantic import BaseModel, Field


class SpacyConfig(BaseModel):
    """spaCy embedding configuration."""

    model: str = "en_core_web_md"
    batch_size: int = Field(default=1024, ge=1)
    n_process: int = -1

    exclude: tuple[str, ...] = (
        "tagger",
        "parser",
        "senter",
        "attribute_ruler",
        "lemmatizer",
        "ner",
    )


class XGBoostTrainingConfig(BaseModel):
    """XGBoost training configuration."""

    n_estimators: int = Field(default=200, ge=1)
    max_depth: int = Field(default=4, ge=1)
    learning_rate: float = Field(default=0.1, gt=0)
    subsample: float = Field(default=0.7, gt=0, le=1)
    colsample_bytree: float = Field(default=0.5, gt=0, le=1)

    tree_method: str = "hist"
    objective: str = "binary:logistic"
    eval_metric: str = "logloss"

    random_state: int = 42
    n_jobs: int = -1


class ThresholdConfig(BaseModel):
    """Classification thresholds."""

    default_threshold: float = 0.5

    min_threshold: float = 0.0
    max_threshold: float = 1.0
    num_thresholds: int = 101

    claim: float = Field(
        default=0.45,
        ge=0,
        le=1,
    )

    evidence: float = Field(
        default=0.35,
        ge=0,
        le=1,
    )


class OptunaConfig(BaseModel):
    """Optuna hyperparameter tuning configuration."""

    n_trials: int = Field(
        default=50,
        ge=1,
    )

    cv_folds: int = Field(
        default=5,
        ge=2,
    )

    n_jobs: int = Field(
        default=4,
    )

    seed: int = 42

    n_estimators_min: int = Field(
        default=100,
        ge=1,
    )

    n_estimators_max: int = Field(
        default=1000,
        ge=1,
    )

    n_estimators_step: int = Field(
        default=100,
        ge=1,
    )

    max_depth_min: int = Field(
        default=3,
        ge=1,
    )

    max_depth_max: int = Field(
        default=10,
        ge=1,
    )

    learning_rate_min: float = Field(
        default=0.01,
        gt=0,
    )

    learning_rate_max: float = Field(
        default=0.3,
        gt=0,
    )

    subsample_min: float = Field(
        default=0.6,
        gt=0,
        le=1,
    )

    subsample_max: float = Field(
        default=1.0,
        gt=0,
        le=1,
    )

    colsample_bytree_min: float = Field(
        default=0.4,
        gt=0,
        le=1,
    )

    colsample_bytree_max: float = Field(
        default=1.0,
        gt=0,
        le=1,
    )


class DatasetConfig(BaseModel):
    """Dataset sampling configuration."""

    claim_proof_of_concept_max_samples: int = Field(
        default=5_000,
        ge=1,
    )

    evidence_proof_of_concept_max_samples: int = Field(
        default=1_000,
        ge=1,
    )


class XGBoostExtractorConfig(BaseModel):
    """Complete XGBoost claim extractor configuration."""

    dataset: DatasetConfig = DatasetConfig()
    spacy: SpacyConfig = SpacyConfig()
    training: XGBoostTrainingConfig = XGBoostTrainingConfig()
    threshold: ThresholdConfig = ThresholdConfig()
    optuna: OptunaConfig = OptunaConfig()
