"""Module with an argument framer factory function."""

from src.argument_framing.argument_framer import ArgumentFramer
from src.argument_framing.argument_framing_dataset import ArgumentFramingDataset
from src.configuration import config
from src.data_models.data_models import ArgumentFramerImplementation


def build_argument_framer(
    framer_implementation: ArgumentFramerImplementation | None = None,
) -> ArgumentFramer:
    """
    Construct a ready-made argument framer from pre-configured implementations.

    Args:
        framer_implementation (ArgumentFramerImplementation): Choice of a
            pre-configured implementation.

    Returns:
        ArgumentFramer: An initialised instance of the argument framer.

    Raises:
        NotImplementedError: Raised if not implemented variant was requested.
    """
    # Local imports for lazy loading prevent importing heavy implementation
    # modules when they are not requested.
    # pylint: disable=import-outside-toplevel
    from src.argument_framing.xgboost_framer import XGBoostFrameClassifier
    # pylint: enable=import-outside-toplevel

    framer_implementation = framer_implementation or config.argument_framer

    match framer_implementation:
        case ArgumentFramerImplementation.XGBOOST:
            return XGBoostFrameClassifier(dataset=ArgumentFramingDataset())
