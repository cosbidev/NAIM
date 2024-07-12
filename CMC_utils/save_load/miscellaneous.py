import os
import yaml
from omegaconf import OmegaConf, DictConfig
from CMC_utils.paths import add_extension

__all__ = ["save_yaml", "load_yaml"]


def save_yaml(cfg: DictConfig, filename: str, path: str, extension: str = None) -> None:
    """
    Saves a DictConfig to a yaml file
    Parameters
    ----------
    cfg : DictConfig
    filename : str
    path : str
    extension : str

    Returns
    -------
    None
    """
    if extension is not None:
        filename = add_extension(filename, extension)
    # dumps to file:
    OmegaConf.save(cfg, os.path.join(path, filename))


def load_yaml(path: str, **kwargs):

    with open(path, 'r') as file:
        content = yaml.safe_load(file)

    return content


if __name__ == "__main__":
    pass
