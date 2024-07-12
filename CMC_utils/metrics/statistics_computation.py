import os
import re
import logging
import pandas as pd
from typing import List, Union
from omegaconf import DictConfig
from CMC_utils import save_load
from CMC_utils.paths import get_files_with_extension
from .performance_computation import *


log = logging.getLogger(__name__)

__all__ = ["compute_performance", "compute_missing_performance"]


def compute_missing_performance(classes: List[str], prediction_path: str, results_path: str, metrics: Union[dict, DictConfig], missing_percentages: List[float]) -> None:
    """
    Compute the performance for different missing percentages
    Parameters
    ----------
    classes : List[str]
    prediction_path : str
    results_path : str
    metrics : Union[dict, DictConfig]
    missing_percentages : List[float]

    Returns
    -------
    None
    """
    missing_percentages = [int(miss_perc * 100) for miss_perc in missing_percentages]

    for train_missing_percentage in missing_percentages:
        preds_path = os.path.join( prediction_path, str(train_missing_percentage) )
        res_path = os.path.join( results_path, str(train_missing_percentage) )

        compute_performance(classes=classes, prediction_path=preds_path, results_path=res_path, metrics=metrics)

        for test_missing_percentage in missing_percentages:
            test_preds_path = os.path.join( preds_path, str(test_missing_percentage) )
            test_res_path = os.path.join(res_path, str(test_missing_percentage))
            compute_performance(classes=classes, prediction_path=test_preds_path, results_path=test_res_path, metrics=metrics)


def compute_performance(classes: List[str], prediction_path: str, results_path: str, metrics: Union[dict, DictConfig]) -> None:
    """
    Compute the performance for a given set of predictions
    Parameters
    ----------
    classes : List[str]
    prediction_path : str
    results_path : str
    metrics : Union[dict, DictConfig]

    Returns
    -------
    None
    """
    prediction_files = get_files_with_extension(prediction_path, "csv")
    prediction_files = sorted(prediction_files, key=lambda file_path: int(re.findall(r"\d+_(\d+)_\w+\.csv\Z", file_path)[0]))
    prediction_files = sorted(prediction_files, key=lambda file_path: int(re.findall(r"(\d+)_\d+_\w+\.csv\Z", file_path)[0]))

    for fset in ["train", "val", "test"]:
        set_files = [file for file in prediction_files if file.endswith(f"{fset}.csv")]

        performance, performance_balanced = pd.DataFrame(), pd.DataFrame()

        for file in set_files:
            test_fold, val_fold = [ int(fold_num) for fold_num in re.findall(r"(\d+)_(\d+)_\w+\.csv\Z", file)[0] ]  # [0]

            fold_preds = save_load.load_params_table(file).set_index("ID")  # , converters={"probability": eval}

            fold_performance = metrics_computation_df(fold_preds, metrics=metrics, classes=classes, use_weights=False, verbose=0).mul(100).round(2).reset_index().assign(set=fset, test_fold=test_fold, val_fold=val_fold)
            fold_performance_balanced = metrics_computation_df(fold_preds, metrics=metrics, classes=classes, use_weights=True, verbose=0).mul(100).round(2).reset_index().assign(set=fset, test_fold=test_fold, val_fold=val_fold)
            performance = pd.concat([ performance, fold_performance ], axis=0, ignore_index=True )
            performance_balanced = pd.concat([ performance_balanced, fold_performance_balanced ], axis=0, ignore_index=True )

        if not performance.empty:
            unbalanced_path = os.path.join(results_path, "unbalanced", fset)
            if not os.path.exists(unbalanced_path):
                os.makedirs(unbalanced_path)
            compute_performance_statistics(performance, unbalanced_path)

        if not performance_balanced.empty:
            balanced_path = os.path.join(results_path, "balanced", fset)
            if not os.path.exists(balanced_path):
                os.makedirs(balanced_path)
            compute_performance_statistics(performance_balanced, balanced_path)


def compute_performance_statistics(performance, path) -> None:
    """
    Compute the performance statistics
    Parameters
    ----------
    performance : pd.DataFrame
    path : str

    Returns
    -------
   None
    """
    save_load.save_table(performance, f"all_test_performance.csv", path, index=False)

    mean_performance = performance.drop(["test_fold", "val_fold"], axis=1).groupby( by=["set", "class" ] ).agg( [ "mean", "std", "min", "max" ] ).round(2)
    save_load.save_table( mean_performance.reset_index(), f"classes_average_performance.csv", path, index=False )

    average_performance = performance.drop(["test_fold", "val_fold", "class"], axis=1).groupby(by=["set"]).agg(["mean", "std", "min", "max"]).round(2)
    save_load.save_table( average_performance.reset_index(), f"set_average_performance.csv", path, index=False )
    log.info("Average performance computed")


if __name__ == "__main__":
    pass
