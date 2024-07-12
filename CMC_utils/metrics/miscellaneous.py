from typing import Union
from hydra.utils import call
from omegaconf import DictConfig

__all__ = ["set_metrics_params"]


def set_metrics_params(metrics: Union[dict, DictConfig], preprocessing_params: dict):
    """
    Set the parameters of the metrics
    Parameters
    ----------
    metrics : Union[dict, DictConfig]
        metrics parameters
    preprocessing_params : dict

    Returns
    -------
    dict
        metrics parameters
    """
    metrics = { key: call(metric_params["set_params_function"], metric_params, preprocessing_params=preprocessing_params, _recursive_=False) for key, metric_params in metrics.items() }
    return metrics


if __name__ == "__main__":
    pass
