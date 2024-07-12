import logging
from typing import Tuple, Union
from hydra.utils import instantiate
from omegaconf import DictConfig
from CMC_utils.miscellaneous import recursive_cfg_search, recursive_cfg_substitute
from CMC_utils.datasets import SupervisedTabularDatasetTorch

__all__ = ["set_adaboost_params", "set_histgradientboostingtree_params", "set_naim_params", "set_tabnet_params", "set_xgboost_params", "set_fttransformer_params", "set_tabtransformer_params"]


log = logging.getLogger(__name__)


def compute_categorical_idxs_dims(columns: list, preprocessing_params: Union[dict, DictConfig]) -> Tuple[list, list]:
    """
    Compute the indexes and dimensions of the categorical features in the dataset
    Parameters
    ----------
    columns : list
    preprocessing_params : Union[dict, DictConfig]

    Returns
    -------
    Tuple[list, list]
        categorical_idxs, categorical_dims
    """
    unique_values = preprocessing_params.categorical_unique_values
    categorical_columns = tuple(unique_values.index.to_list())

    categorical_idxs = []
    categorical_dims = []
    for idx, col in enumerate(columns):
        if col in categorical_columns:
            categorical_idxs.append(idx)
            categorical_dims.append(len(unique_values[col]))
        elif col.split("_")[0] in categorical_columns:
            categorical_idxs.append(idx)
            categorical_dims.append(2)
    return categorical_idxs, categorical_dims


def set_adaboost_params(model_cfg: dict, **_) -> dict:
    """
    Set the parameters of the HistGradientBoostingTree model
    Parameters
    ----------
    model_cfg : dict

    Returns
    -------
    dict
        model_cfg
    """
    if model_cfg["init_params"].get("estimator", None) is not None:
        model_cfg["init_params"]["estimator"] = instantiate(model_cfg["init_params"]["estimator"])

    return model_cfg


def set_histgradientboostingtree_params(model_cfg: dict, preprocessing_params: Union[dict, DictConfig], train_set: SupervisedTabularDatasetTorch, **_) -> dict:
    """
    Set the parameters of the HistGradientBoostingTree model
    Parameters
    ----------
    model_cfg : dict
    preprocessing_params : Union[dict, DictConfig]
    train_set : SupervisedTabularDatasetTorch

    Returns
    -------
    dict
        model_cfg
    """
    model_cfg["init_params"]["categorical_features"] = compute_categorical_idxs_dims(train_set.columns, preprocessing_params)[0]

    return model_cfg


def set_naim_params(model_cfg: dict, preprocessing_params: Union[dict, DictConfig], train_set: SupervisedTabularDatasetTorch, **_) -> dict:
    """
    Set the parameters of the NAIM model
    Parameters
    ----------
    model_cfg : dict
    preprocessing_params : Union[dict, DictConfig]
    train_set : SupervisedTabularDatasetTorch

    Returns
    -------
    dict
        model_cfg
    """
    model_cfg["init_params"]["input_size"] = train_set.input_size
    model_cfg["init_params"]["output_size"] = train_set.output_size

    cat_idxs, cat_dims = compute_categorical_idxs_dims(train_set.columns, preprocessing_params)
    model_cfg["init_params"]["cat_idxs"] = cat_idxs
    model_cfg["init_params"]["cat_dims"] = cat_dims

    searched_value, key_found = recursive_cfg_search(model_cfg, "d_token")
    if searched_value is None and key_found:
        d_token, _ = recursive_cfg_search(model_cfg, "input_size")
        log.info(f"Token dimension: {d_token}")
        model_cfg = recursive_cfg_substitute(model_cfg, {"d_token": d_token})

    searched_value, key_found = recursive_cfg_search(model_cfg, "num_heads")
    if searched_value is None and key_found:
        divisor = 2
        log.info(f"N. attention heads: {divisor}")
        model_cfg = recursive_cfg_substitute(model_cfg, {"num_heads": divisor})

    return model_cfg


def set_tabnet_params(model_cfg: dict, preprocessing_params: Union[dict, DictConfig], train_set: SupervisedTabularDatasetTorch, val_set: SupervisedTabularDatasetTorch, **_) -> dict:
    """
    Set the parameters of the TabNet model
    Parameters
    ----------
    model_cfg : dict
    preprocessing_params : Union[dict, DictConfig]
    train_set : SupervisedTabularDatasetTorch
    val_set : SupervisedTabularDatasetTorch

    Returns
    -------
    dict
        model_cfg
    """
    cat_idxs, cat_dims = compute_categorical_idxs_dims(train_set.columns, preprocessing_params)
    model_cfg["init_params"]["cat_idxs"] = cat_idxs
    model_cfg["init_params"]["cat_dims"] = cat_dims

    model_cfg["fit_params"]["eval_set"] = [train_set.get_data()[:-1], val_set.get_data()[:-1]]
    model_cfg["fit_params"]["eval_name"] = ["train", "val"]

    return model_cfg


def set_xgboost_params(model_cfg: dict, preprocessing_params: Union[dict, DictConfig], train_set: SupervisedTabularDatasetTorch, val_set: SupervisedTabularDatasetTorch, **_) -> dict:
    """
    Set the parameters of the XGBoost model
    Parameters
    ----------
    model_cfg : dict
    preprocessing_params : Union[dict, DictConfig]
    train_set : SupervisedTabularDatasetTorch
    val_set : SupervisedTabularDatasetTorch

    Returns
    -------
    dict
        model_cfg
    """
    cat_idxs = compute_categorical_idxs_dims(train_set.columns, preprocessing_params)[0]
    model_cfg["init_params"]["feature_types"] = list("c" if n in cat_idxs else "q" for n in range(train_set.input_size))
    model_cfg["init_params"]["num_class"] = len(preprocessing_params["classes"])
    model_cfg["fit_params"]["eval_set"] = [train_set.get_data()[:-1], val_set.get_data()[:-1]]

    return model_cfg

def set_fttransformer_params(model_cfg: dict, preprocessing_params: Union[dict, DictConfig], train_set: SupervisedTabularDatasetTorch, **_) -> dict:
    cat_idxs, cat_dims = compute_categorical_idxs_dims(train_set.columns, preprocessing_params)
    model_cfg["init_params"]["cat_idxs"] = cat_idxs
    model_cfg["init_params"]["categories"] = cat_dims
    model_cfg["init_params"]["num_continuous"] = len(preprocessing_params["numerical_columns"])
    model_cfg["init_params"]["dim_out"] = len(preprocessing_params["classes"])
    return model_cfg


def set_tabtransformer_params(model_cfg: dict, preprocessing_params: Union[dict, DictConfig], train_set: SupervisedTabularDatasetTorch, **_) -> dict:
    cat_idxs, cat_dims = compute_categorical_idxs_dims(train_set.columns, preprocessing_params)
    model_cfg["init_params"]["cat_idxs"] = cat_idxs
    model_cfg["init_params"]["categories"] = cat_dims
    model_cfg["init_params"]["num_continuous"] = len(preprocessing_params["numerical_columns"])
    model_cfg["init_params"]["dim_out"] = len(preprocessing_params["classes"])
    return model_cfg


if __name__ == "__main__":
    pass
