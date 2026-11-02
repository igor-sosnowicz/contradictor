"""Module with utility functions for the MLP encoder."""

import torch

from src.data_models.data_models import StyledFramedArgument


def cosine_distance(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """
    Calculate cosine distances between two batches of vectors.

    Args:
        first (torch.Tensor): The first batch of vectors.
        second (torch.Tensor): The second batch of vectors.

    Returns:
        torch.Tensor: Cosine distances between two batches.
    """
    first_norm = torch.nn.functional.normalize(first, p=2, dim=1)
    second_norm = torch.nn.functional.normalize(second, p=2, dim=1)
    cosine_sim = torch.sum(first_norm * second_norm, dim=1)
    return 1.0 - cosine_sim


def row_to_input_tensor(row: dict, suffix: str) -> torch.Tensor:
    """Convert a DataFrame row into a model input tensor."""
    return StyledFramedArgument.from_row(row, suffix).to_tensor()


def mlp_encoder_filename(input_dim: int, output_dim: int) -> str:
    """
    Build the filename for an encoder saved with input/output dimensionality.

    Args:
        input_dim (int): A number of features.
        output_dim (int): A number of outputs.

    Returns:
        str: Filename for a model of given dimensionality.
    """
    return f"mlp_encoder_in{input_dim}_out{output_dim}.pkl"
