import logging
import numpy as np
import pandas as pd
from hydra.utils import call
from omegaconf import OmegaConf
from CMC_utils.miscellaneous import recursive_cfg_substitute
from sklearn.utils.class_weight import compute_sample_weight

log = logging.getLogger(__name__)

__all__ = ["metrics_computation", "metrics_computation_df"]


def metrics_computation_df(data, **kwargs) -> pd.DataFrame:
    """
    Compute the metrics from a dataframe
    Parameters
    ----------
    data : pd.DataFrame
    kwargs : dict

    Returns
    -------
    pd.DataFrame
        metrics
    """
    labels = np.vstack(data.label.values)
    predictions = np.vstack(data.prediction.values)
    probabilities = np.vstack(data.probability.values)

    return metrics_computation(labels, predictions, probabilities, **kwargs)


def metrics_computation(label, prediction, probability, metrics: dict, round_number=4, verbose: bool = True, use_weights: bool = False, **kwargs) -> pd.DataFrame:
    """
    Compute the metrics
    Parameters
    ----------
    label : np.ndarray
    prediction : np.ndarray
    probability : np.ndarray
    metrics : dict
    round_number : int
    verbose : bool
    use_weights : bool
    kwargs : dict

    Returns
    -------
    pd.DataFrame
    """
    classes = kwargs.get("classes", np.sort(np.unique(label)))
    classes_map = {int(v): str(k) for v, k in enumerate(classes)}
    classes_map_inverted = {v: k for k, v in classes_map.items()}

    if label.dtype not in (int, float):
        vectorized_func = np.vectorize(lambda x: classes_map_inverted.get(str(x), None))
        label = vectorized_func(label)
        prediction = vectorized_func(prediction)

    if use_weights:
        sample_weight = compute_sample_weight(class_weight='balanced', y=label)
        metrics = OmegaConf.create(metrics) if type(metrics) == dict else metrics
        metrics = OmegaConf.to_object(metrics)
        metrics = recursive_cfg_substitute(metrics, dict(sample_weight=sample_weight))

    is_binary = len(probability[0]) == 2 and np.round(sum(probability[0]), 0) == 1

    performance = dict()
    for metric in metrics.values():
        if is_binary and metric["name"] == "auc":
            prob = probability[:, 1]
        else:
            prob = probability

        perf = call( metric["init"], y_true=label, y_pred=prediction, y_score=prob, labels=list(classes_map.keys()))

        performance[metric["name"]] = np.round(perf, round_number)

    performance = pd.DataFrame(performance, index=classes).rename_axis("class", axis=0)

    if verbose:
        log.info("Performance computed")

    return performance


if __name__ == "__main__":
    pass
