import yaml
import logging
from pathlib import Path
from typing import Dict, Any
from .logging_configs import setup_logging


setup_logging()


def load_config(config_path:  Path) -> Dict[str, Any]:
    """
    Loads a YAML configuration file and returns its contents as a dictionary.

    This function ensures that:
    - The file exists before attempting to read it.
    - The file is a valid YAML format.
    - Any YAML parsing errors are handled gracefully.

    Args:
        config_path (Path): Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: A dictionary containing configuration parameters.

    Raises:
        FileNotFoundError: If the specified configuration file does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.
        Exception: If an unexpected error occurs.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        logging.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}

    except yaml.YAMLError as exc:
        logging.error(f"Error parsing YAML file {config_path}: {exc}")
        raise

    except Exception as exc:
        logging.error(f"Unexpected error while loading config file {config_path}: {exc}")
        raise
