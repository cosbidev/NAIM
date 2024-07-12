import os
import logging
from logging.handlers import RotatingFileHandler
from omegaconf import DictConfig
from CMC_utils import miscellaneous, save_load

log = logging.getLogger(__name__)

__all__ = ["initialize_experiment"]


def initialize_experiment(cfg: DictConfig) -> None:
    """
    Initialize experiment by setting seed and creating directories
    Parameters
    ----------
    cfg : DictConfig

    Returns
    -------
    None
    """
    miscellaneous.seed_all(cfg.seed)

    save_load.create_directories(allow_existing=cfg.continue_experiment, **cfg.paths)

    setup_new_log_file(cfg.paths.experiment, cfg.experiment_name)

    defaults_paths_cfg = save_load.load_yaml("./confs/experiment/paths/default.yaml")

    complete_cfg = miscellaneous.join_dictionaries(cfg, defaults_paths_cfg)

    save_load.save_yaml(complete_cfg, "config", cfg.paths.experiment, "yaml")


def setup_new_log_file(log_directory, log_file_name):

    # Define the full log path
    full_log_path = os.path.join(log_directory, log_file_name + ".log")

    # Get the root logger
    logger = logging.getLogger()

    # Remove existing file handlers
    for handler in logger.handlers[:]:
        if isinstance(handler, (logging.FileHandler, RotatingFileHandler)):
            logger.removeHandler(handler)

    # Create a new file handler and add it to the logger
    file_handler = logging.FileHandler(full_log_path)
    logger.addHandler(file_handler)

    # Optionally, set the level and formatter for the new handler
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] - [%(levelname)s] - %(message)s - [%(name)s]')
    file_handler.setFormatter(formatter)


if __name__ == "__main__":
    pass
