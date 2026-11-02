"""Module with re-usable project-wide Pydantic validators."""

import torch


def validate_device_presence(value: str) -> str:
    """
    Validate if a string contains the name of a valid and available device accelerator.

    Args:
        value (str): The name of a device. Usually: "cpu", "cuda", "cuda:0", "mps",
            "xpu", "xla", "meta". For the full list check torch's website:
            https://docs.pytorch.org/docs/2.15/tensor_attributes.html#torch.device

    Returns:
        str: A valid device name.

    Raises:
        ValueError: Raised if either an invalid name or a name of
            unavailable device was used.
    """
    try:
        torch.device(value)
    except RuntimeError as e:
        raise ValueError(
            f"The device {value} is either unavailable or not "
            "a correct acceleration device."
        ) from e

    return value
