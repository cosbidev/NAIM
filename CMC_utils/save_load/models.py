import os
import pickle
import torch.nn
import pandas as pd
from typing import Union
from .tables import save_table
from omegaconf import DictConfig
from CMC_utils.paths import add_extension, get_extension, remove_extension


__all__ = ["load_model", "save_model", "save_params"]

## load

def _load_pkl( path: str ) -> object:
    """
    Loads a pickle file from the specified file path.

    Parameters:
        path : str.
            The file path of the pickle file to be loaded.

    Returns:

    model : object.
        The loaded model from the pickle file.
    """
    model = pickle.load( open( path, 'rb' ) )
    return model


def _load_torch_model( path: str ) -> torch.nn.Module:
    """
    Loads a PyTorch model from the specified file path.

    Parameters:
        path : str.
            The file path of the PyTorch model to be loaded.

    Returns:

    model : torch.nn.Module.
        The loaded PyTorch model.
    """
    model = torch.load( path )
    return model


def load_model(path: str) -> Union[object, torch.nn.Module]:
    """
    Loads a model from the specified file path. The function automatically detects the file type and uses the appropriate function to load the model.

    Parameters:
        path: str.
            The file path of the model to be loaded.

    Returns:

    model: object | torch.nn.Module.
        The loaded model.
    """
    extension = get_extension(path).lower()

    load_options = { "pt": _load_torch_model, "pth": _load_torch_model, "pkl": _load_pkl }

    model = load_options[ extension ](path)

    return model

## save

def _save_pkl( model: object, filename: str, path: str, **_ ) -> None:
    """
    Saves a model in pickle format.

    Parameters:
        model : object.
            The model to be saved.
        filename : str.
            The desired file name for the saved model.
        path : str.
            The directory path where the model will be saved.

    Returns:

    None
    """
    path = os.path.join( path, filename )
    pickle.dump( model, open( path, 'wb' ) )

def _save_torch_model( model, filename: str, path: str, **_ ) -> None:
    """
    Saves a model in torch format.

    Parameters:
        model : object.
            The model to be saved.
        filename : str.
            The desired file name for the saved model.
        path : str.
            The directory path where the model will be saved.
    """
    path = os.path.join( path, filename )

    torch.save( model, path )


def save_model(model: Union[ object, torch.nn.Module ], model_name: str, model_dir: str, extension: str = None, model_params: Union[dict, DictConfig, pd.Series, pd.DataFrame] = None) -> None:
    """
    Saves a model in the format specified by the file extension.

    Parameters:
        model : object.
            The model to be saved.
        model_name : str.
            The desired file name for the saved model.
        model_dir : str.
            The directory path where the model will be saved.
        extension : str.
            Extension of the desired model.
        model_params : Union[dict, pd.Series, pd.DataFrame].
            Additional parameters associated with the model. If provided, they will be saved in a separate csv file.

    Returns:

    None

    """
    if extension is not None:
        model_name = add_extension(model_name, extension)
    else:
        extension = get_extension(model_name).lower()

    assert extension is not None, "Model extension not provided"

    save_options = { "pt": _save_torch_model, "pth": _save_torch_model, "pkl": _save_pkl }

    save_options[ extension ](model, model_name, model_dir)

    if model_params is not None:
        model_name = remove_extension(model_name)
        save_params(model_params, model_name, model_dir, extension="csv")


def save_params(params: dict, filename: str, path: str, extension: str = "csv", **kwargs) -> None:
    """
    Saves a dictionary of parameters in a csv file.
    Parameters
    ----------
    params : dict
    filename : str
    path : str
    extension : str
    kwargs : dict

    Returns
    -------
    None
    """
    model_params_df = pd.DataFrame( pd.Series( params ).reset_index() ).rename( {"index": "params", 0: "values"}, axis=1 )
    save_table(model_params_df, filename, path, extension=extension, **kwargs)


if __name__ == "__main__":
    pass
