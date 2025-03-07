import yaml
import logging
from pathlib import Path
from typing import Dict, Any
from hydra import initialize, compose
from omegaconf import OmegaConf, DictConfig
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


def load_hydra_config(config_dir: Path, config_name: str) -> DictConfig:
    """
    Loads a Hydra configuration using OmegaConf and returns its contents as a DictConfig.

    Args:
        config_dir (Path): The directory containing the Hydra configuration files.
        config_name (str): The base name of the configuration file (without the .yaml extension).

    Returns:
        DictConfig: A DictConfig object representing the Hydra configuration.

    Raises:
        Exception: If any error occurs during configuration loading or composition.
    """
    try:
        with initialize(config_path=str(config_dir)):
            cfg = compose(config_name=config_name)
            logging.info("Hydra configuration loaded successfully.")
            return cfg

    except Exception as exc:
        logging.error(f"Error loading Hydra configuration from {config_dir} with config name {config_name}: {exc}")
        raise

