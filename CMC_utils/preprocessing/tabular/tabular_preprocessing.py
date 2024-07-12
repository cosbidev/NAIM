import os
import logging
import numpy as np
import pandas as pd
from typing import Union, List
from omegaconf import DictConfig

from .functions import *
from CMC_utils import save_load

log = logging.getLogger(__name__)

__all__ = ["set_fold_preprocessing", "apply_preprocessing", "describe_features"]


def set_fold_preprocessing(train, test_fold: int, val_fold: int, preprocessing_paths: Union[dict, DictConfig], preprocessing_params: dict, **kwargs) -> None:
    """
    Applies preprocessing to a fold and saves the parameters
    Parameters
    ----------
    train : pd.DataFrame
    test_fold : int
    val_fold : int
    preprocessing_paths : Union[dict, DictConfig]
    preprocessing_params : dict
    kwargs : dict

    Returns
    -------
    None
    """
    train = train.copy()
    filename_wo_extension = f"{test_fold}_{val_fold}"

    if "categorical" in preprocessing_params.keys() and preprocessing_params["categorical_columns"]:
        if not os.path.exists( os.path.join( preprocessing_paths["categorical"], filename_wo_extension + ".csv")):

            train, train_categorical_params = categorical_preprocessing(train, unique_values=preprocessing_params["categorical_unique_values"], return_categories=True, **preprocessing_params["categorical"], **kwargs)

            save_load.save_table(pd.Series(train_categorical_params).reset_index().rename( {"index": "params", 0: "values"}, axis=1 ), filename_wo_extension, preprocessing_paths["categorical"], extension="csv")
            log.info("Categorical preprocessing params saved")

    if "numerical" in preprocessing_params.keys():
        if not os.path.exists(os.path.join(preprocessing_paths["numerical"], filename_wo_extension + ".csv")):

            if preprocessing_params["numerical"]["all_columns"]:
                columns_to_preprocess = list(preprocessing_params["numerical_columns"].values()) + list(preprocessing_params["categorical_columns"].values())
                columns_to_preprocess.remove("ID")
            else:
                columns_to_preprocess = list(preprocessing_params["numerical_columns"].values())

            train, train_numerical_params = numerical_preprocessing(train, numerical_columns=columns_to_preprocess, return_params=True, **preprocessing_params["numerical"])

            save_load.save_table(pd.Series(train_numerical_params).reset_index().rename( {"index": "params", 0: "values"}, axis=1 ), filename_wo_extension, preprocessing_paths["numerical"], extension="csv")
            log.info("Numerical preprocessing params saved")

    if "imputer" in preprocessing_params.keys():
        if not os.path.exists(os.path.join(preprocessing_paths["imputer"], filename_wo_extension + ".pkl")):
            train, train_imputer = impute_missing(train, features_info=preprocessing_params, return_imputer=True, **preprocessing_params["imputer"])

            save_load.save_model(train_imputer, filename_wo_extension, preprocessing_paths["imputer"], extension="pkl", model_params=pd.Series(preprocessing_params["imputer"]))
            log.info("Imputation model saved")


def apply_preprocessing(*sets: pd.DataFrame, preprocessing_paths: Union[dict, DictConfig], preprocessing_params: dict, test_fold: int = 0, val_fold: int = 0, copy: bool = True, **kwargs ) -> List[pd.DataFrame]:
    """
    Applies preprocessing to a fold
    Parameters
    ----------
    sets : pd.DataFrame
    preprocessing_paths : Union[dict, DictConfig]
    preprocessing_params : dict
    test_fold : int (default: 0)
    val_fold : int (default: 0)
    copy : bool (default: True)
    kwargs : dict

    Returns
    -------
    List[pd.DataFrame]
    """
    if copy:
        sets = [fset.copy() for fset in sets]
    filename_wo_extension = f"{test_fold}_{val_fold}"

    categorical_path = os.path.join( preprocessing_paths["categorical"], filename_wo_extension + ".csv" )
    if os.path.exists(categorical_path):
        categorical_params = save_load.load_params_table(categorical_path, index_col=0).squeeze().to_dict()
        sets = [ categorical_preprocessing(fset, **categorical_params) for fset in sets ]

    numerical_path = os.path.join(preprocessing_paths["numerical"], filename_wo_extension + ".csv")
    if os.path.exists(numerical_path):
        numerical_params = save_load.load_params_table(numerical_path, index_col=0).squeeze().to_dict()
        sets = [ numerical_preprocessing(fset, **numerical_params) for fset in sets]

    imputer_path = os.path.join(preprocessing_paths["imputer"], filename_wo_extension + ".pkl")
    if os.path.exists(imputer_path):
        imputer = save_load.load_model(imputer_path)
        imputer_params = save_load.load_params_table( os.path.join( preprocessing_paths["imputer"], filename_wo_extension + ".csv" ), index_col=0).squeeze().to_dict()
        sets = [ impute_missing(fset, features_info=preprocessing_params, imputer=imputer, **imputer_params) for fset in sets]

    return sets


def describe_features(data: pd.DataFrame, features_dtypes: pd.Series) -> pd.DataFrame:
    features = pd.DataFrame()
    n_samples = data.shape[0]
    for feature, dtype in features_dtypes.items():

        nan_number = data[feature].isna().sum()
        feature_info = {"Feature": feature}

        if dtype in ["float", "int"]:
            mean = np.round(data[feature].mean(), 2)

            feature_info["Categories"] = [f"<{mean}", f"≥{mean}"]

            le_num = np.sum(data[feature] < mean)
            ge_num = np.sum(data[feature] >= mean)
            feature_info["Distribution"] = [ f"{le_num} ({np.round(100*(le_num/n_samples), 2)}%)", f"{ge_num} ({np.round(100*(ge_num/n_samples), 2)}%)" ]

        else:
            feat_info = data[feature].value_counts().sort_index().to_dict()
            feature_info["Categories"] = list(feat_info.keys())
            feature_info["Distribution"] = [ f"{v} ({np.round(100*(v/n_samples), 2)}%)" for v in list(feat_info.values())]

        feature_info = pd.DataFrame(feature_info)
        nan_col = pd.Series( [f"{nan_number} ({np.round(100 * (nan_number / n_samples), 2)}%)"] + [np.nan] * (len(feature_info["Categories"]) - 1))
        feature_info.insert(1, "Missing Data", nan_col )
        features = pd.concat([features, feature_info], axis=0, ignore_index=True)

    return features


if __name__ == "__main__":
    pass
