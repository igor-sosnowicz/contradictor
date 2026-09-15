"""A module being an entrypoint to the entire application."""

from loguru import logger


def main() -> None:
    """Run the application."""
    logger.info("Hello from contradictor!")


if __name__ == "__main__":
    main()
