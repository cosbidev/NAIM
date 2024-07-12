import os
import pandas as pd
from kan import KAN
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from typing import Union
from omegaconf import DictConfig
from CMC_utils.save_load import save_table
from CMC_utils.paths import add_extension, get_extension, remove_extension


__all__ = ["save_tabnet_model", "load_tabnet_model", "save_xgboost_model", "load_xgboost_model", "save_kan_model", "load_kan_model"]


def save_xgboost_model(model: XGBClassifier, model_name: str, model_dir: str, extension: str = None, model_params: Union[dict, DictConfig, pd.Series, pd.DataFrame] = None, **kwargs) -> None:
    """
    Save XGBoost model
    Parameters
    ----------
    model : XGBClassifier
    model_name : str
    model_dir : str
    extension : str
    model_params : Union[dict, DictConfig, pd.Series, pd.DataFrame]
    kwargs : dict

    Returns
    -------
    None
    """
    if extension is not None:
        model_name = add_extension(model_name, extension)
    else:
        extension = get_extension(model_name).lower()

    assert extension is not None, "Model extension not provided"

    model.save_model(os.path.join( model_dir, model_name))

    if model_params is not None:
        model_params_df = pd.DataFrame(pd.Series(model_params).reset_index()).rename({"index": "params", 0: "values"}, axis=1)
        model_name = remove_extension(model_name)

        save_table(model_params_df, model_name, model_dir, extension="csv")


def load_xgboost_model(path: str) -> XGBClassifier:
    """
    Load XGBoost model
    Parameters
    ----------
    path : str

    Returns
    -------
    XGBClassifier
    """
    model = XGBClassifier()
    model.load_model(path)
    return model


def save_tabnet_model(model: TabNetClassifier, model_name: str, model_dir: str, extension: str = None, model_params: Union[dict, DictConfig, pd.Series, pd.DataFrame] = None) -> None:
    """
    Save TabNet model
    Parameters
    ----------
    model : TabNetClassifier
    model_name : str
    model_dir : str
    extension : str
    model_params : Union[dict, DictConfig, pd.Series, pd.DataFrame]

    Returns
    -------
    None
    """
    if extension is not None:
        model_name = add_extension(model_name, extension)
    else:
        extension = get_extension(model_name).lower()

    assert extension is not None, "Model extension not provided"

    model.save_model(os.path.join(model_dir, model_name))

    if model_params is not None:
        model_params_df = pd.DataFrame(pd.Series(model_params).reset_index()).rename({"index": "params", 0: "values"}, axis=1)
        model_name = remove_extension(model_name)

        save_table(model_params_df, model_name, model_dir, extension="csv")


def load_tabnet_model(path: str) -> TabNetClassifier:
    """
    Load TabNet model
    Parameters
    ----------
    path : str
        Model path

    Returns
    -------
    TabNetClassifier
    """
    model = TabNetClassifier()
    model.load_model(path)
    return model


def save_kan_model(model: KAN, model_name: str, model_dir: str, extension: str = None, model_params: Union[dict, DictConfig, pd.Series, pd.DataFrame] = None, **kwargs):

    model.save_ckpt(model_name, model_dir)

    if model_params is not None:
        model_params_df = pd.DataFrame(pd.Series(model_params).reset_index()).rename({"index": "params", 0: "values"}, axis=1)
        model_name = remove_extension(model_name)

        save_table(model_params_df, model_name, model_dir, extension="csv")


def load_kan_model(path: str, model: KAN) -> KAN:
    model_name = os.path.basename(path)
    model_dir = os.path.dirname(path)

    model.load_ckpt(model_name, model_dir)
    return model


if __name__ == "__main__":
    pass
