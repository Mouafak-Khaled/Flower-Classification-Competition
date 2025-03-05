import logging

def setup_logging():
    """Configures logging settings for the project."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
