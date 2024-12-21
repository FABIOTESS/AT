# utils/helpers.py

import yaml
import logging
from logging import handlers

def load_yaml_config(config_file):
    """
    Loads a YAML configuration file.

    Parameters:
    - config_file (str): Path to the YAML file.

    Returns:
    - config (dict): Parsed YAML configuration.
    """
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        if not isinstance(config, dict):
            raise ValueError("Loaded YAML content is not a dictionary.")
        return config
    except Exception as e:
        logging.error(f"Error loading YAML configuration from {config_file}: {e}")
        raise

def setup_logger(name, log_file, level=logging.INFO):
    """
    Sets up a logger with the specified name and log file.

    Parameters:
    - name (str): Name of the logger.
    - log_file (str): File path to save logs.
    - level (int): Logging level.

    Returns:
    - logger (logging.Logger): Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding multiple handlers to the same logger
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

        # Rotating File Handler
        handler = handlers.RotatingFileHandler(
            log_file, maxBytes=5*1024*1024, backupCount=5
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
