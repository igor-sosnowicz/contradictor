"""Module with framing configuration."""

from typing import Self

from pydantic import BaseModel, Field, model_validator


class XGBoostConfig(BaseModel):
    """XGBoost-based framer configuration."""

    # MyPy flags the below attributes when a default is passed as a positional
    # parameters instead of keyword. That's why `default` was used.
    xgb_n_estimators: int = Field(default=30, ge=2)
    xgb_n_jobs: int = Field(default=-1, ge=-1)
    tf_idf_max_features: int = Field(default=10_00, ge=1)
    tf_idf_shortest_n_gram: int = Field(default=1, ge=1)
    tf_idf_longest_n_gram: int = Field(default=2, ge=1)
    random_seed: int | None = None
    optuna_trials: int = Field(default=50, ge=2)
    cross_validation_folds: int = Field(default=5, ge=2)
    cross_validation_metric: str = "f1_macro"

    @model_validator(mode="after")
    def check_n_grams_bounds(self) -> Self:
        """Check if n-gram bounds are valid."""
        if self.tf_idf_shortest_n_gram > self.tf_idf_longest_n_gram:
            raise ValueError(
                "The n-gram range's lower bound cannot be higher than its upper bound."
            )
        return self
