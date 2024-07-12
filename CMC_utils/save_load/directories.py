import os
import shutil
import logging
from omegaconf import DictConfig
from CMC_utils.miscellaneous import do_really_nothing

__all__ = ["create_directory", "delete_directory", "create_directories"]

log = logging.getLogger(__name__)


def create_directory(path: str, allow_existing: bool = True) -> None:
    """
    Create a directory in the specified path. If the directory already exists, it is deleted and created again.
    Parameters
    ----------
    path : str
    allow_existing : bool (default: True) If True, the directory is not deleted if it already exists.

    Returns
    -------
    None
    """
    delete_options = {False: do_really_nothing, True: delete_directory}
    delete_options[ not allow_existing and os.path.exists(path) ](path)
    os.makedirs(path, exist_ok = allow_existing)


def delete_directory(path: str) -> None:
    """
    Delete a directory in the specified path.
    Parameters
    ----------
    path : str

    Returns
    -------
    None
    """
    error_handler = lambda func, path_, exc_info: log.warning(
        f"The following exception occurred: {exc_info}")

    shutil.rmtree(path, ignore_errors=False, onerror=error_handler)


def create_directories(allow_existing: bool = True, **paths_dict: str) -> None:
    """
    Create directories in the specified paths. If the directory already exists, it is deleted and created again.
    Parameters
    ----------
    allow_existing : bool (default: True) If True, the directory is not deleted if it already exists.
    paths_dict : str

    Returns
    -------
    None
    """
    for path in paths_dict.values():
        if isinstance(path, (dict, DictConfig)):
            create_directories(allow_existing=allow_existing, **path)
        else:
            create_directory(path, allow_existing=allow_existing)
    # log.info("Experiment folders created")


if __name__ == "__main__":
    pass
