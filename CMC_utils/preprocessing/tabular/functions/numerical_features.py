import logging
import numpy as np
import pandas as pd
from typing import Union, Tuple

log = logging.getLogger(__name__)

__all__ = ["numerical_preprocessing"]

def normalize(table: pd.DataFrame, axis: int = 0, max_: np.ndarray = None, min_: np.ndarray = None, numerical_columns: list = None, return_params: bool = False, **kwargs) -> Union[ pd.DataFrame, Tuple[pd.DataFrame, dict] ]:
    """
    Normalizes a Pandas DataFrame along a specified axis.

    Parameters:
        table: pd.DataFrame.
            The DataFrame to normalize.
        axis: int, default = 0.
            The axis along which to normalize the DataFrame. Must be 0 or 1.
        max_: np.ndarray, optional.
            The max of the table along the specified axis. If not provided, it is computed from the DataFrame.
        min_: np.ndarray, optional.
            The min of the table along the specified axis. If not provided, it is computed from the DataFrame.
        numerical_columns: list, optional.
            A list of columns to apply the preprocessing to. If not provided, the preprocessing is applied to all columns.
        return_params: bool, default = False.
            Whether to return the preprocessing parameters in addition to the preprocessed DataFrame. Default is False.
        kwargs:
            Additional keyword arguments added for compatibility.

    Returns:

    table: pd.DataFrame.
        The normalized DataFrame.
    parameters: dict, optional.
        If return_params is True, the function returns a tuple containing the normalized DataFrame and the preprocessing parameters.
    """
    # If columns is not provided, apply the preprocessing to all columns
    if numerical_columns is None:
        numerical_columns = table.columns.tolist()

    data = table[ numerical_columns ].values

    # If max is not provided, compute it from the DataFrame
    if max_ is None:
        max_ = np.nanmax( data, axis=axis, keepdims=True)
    elif isinstance(max_, str):
        max_ = np.fromstring(max_)
    elif isinstance(max_, list):
        max_ = np.array(max_)
    # If min is not provided, compute it from the DataFrame
    if min_ is None:
        min_ = np.nanmin( data, axis=axis, keepdims=True)
    elif isinstance(min_, str):
        min_ = np.fromstring(min_)
    elif isinstance(min_, list):
        min_ = np.array(min_)
    # Normalize the DataFrame by subtracting the min and dividing by the difference between max and min
    den = max_ - min_
    den[ den == 0] = 1
    table.loc[:, numerical_columns] = (data - min_) / den  # (max_ - min_ )

    # If return_params is True, return a tuple containing the normalized table and the preprocessing parameters
    # Otherwise, return just the normalized table
    parameters = dict(method="normalize", numerical_columns=numerical_columns, max_=np.array2string(max_, separator=', '), min_=np.array2string(min_, separator=', '), axis=axis)
    return_options = {False: table, True: (table, parameters)}
    return return_options[return_params]


def standardize(table: pd.DataFrame, axis: int = 0, mean_: np.ndarray = None, std_: np.ndarray = None, numerical_columns: list = None, return_params: bool = False, **kwargs) -> Union[ pd.DataFrame, Tuple[pd.DataFrame, dict] ]:
    """
    Standardizes a Pandas DataFrame along a specified axis.

    Parameters:
        table: pd.DataFrame.
            The DataFrame to standardize.
        axis: int, default = 0.
            The axis along which to standardize the DataFrame. Must be 0 or 1.
        eps: float, default = 1e-8.
            A small value added to the standard deviation to avoid division by zero.
        mean_: np.ndarray, optional.
            The mean of the DataFrame along the specified axis. If not provided, it is computed from the DataFrame.
        std_: np.ndarray, optional.
            The standard deviation of the DataFrame along the specified axis. If not provided, it is computed from the DataFrame.
        numerical_columns: list, optional.
            A list of columns to apply the preprocessing to. If not provided, the preprocessing is applied to all columns.
        return_params: bool, default = False.
            Whether to return the preprocessing parameters in addition to the preprocessed DataFrame.
        kwargs:
            Additional keyword arguments added for compatibility.

    Returns:

    table: pd.DataFrame.
            The standardized DataFrame.
    parameters: dict, optional.
            If return_params is True, returns a tuple containing the standardized DataFrame and the preprocessing parameters.
    """
    # If columns is not provided, apply the preprocessing to all columns
    if numerical_columns is None:
        numerical_columns = table.columns.tolist()

    data = table[ numerical_columns ].astype(float).values

    # If mean is not provided, compute it from the DataFrame
    if mean_ is None:
        mean_ = np.nanmean( data, axis=axis, keepdims=True)
    elif isinstance(mean_, str):
        mean_ = np.fromstring(mean_)
    elif isinstance(mean_, list):
        mean_ = np.array(mean_)
    # If std is not provided, compute it from the DataFrame
    if std_ is None:
        std_ = np.nanstd( data, axis=axis, keepdims=True)
    elif isinstance(std_, str):
        std_ = np.fromstring(std_)
    elif isinstance(std_, list):
        std_ = np.array(std_)
    # Standardize the DataFrame by subtracting the mean and dividing by the std
    den = std_
    den[ den == 0 ] = 1
    table.loc[:, numerical_columns] = (data - mean_) / den  # ( std_ + eps )

    # If return_params is True, return a tuple containing the standardized table and the preprocessing parameters
    # Otherwise, return just the standardized table
    parameters = dict(method="standardize", numerical_columns=numerical_columns, mean_= np.array2string(mean_, separator=', '), std_= np.array2string(std_, separator=', '), axis=axis)
    return_options = {False: table, True: (table, parameters)}
    return return_options[return_params]


def numerical_preprocessing(table: pd.DataFrame, method: str = "normalize", **kwargs) -> Union[ pd.DataFrame, Tuple[pd.DataFrame, dict] ]:
    """
    Applies numerical preprocessing to a Pandas DataFrame.

    Parameters:
        table: pd.DataFrame.
            The DataFrame to preprocess.
        method: str, default = "normalize".
            The method to use for preprocessing. Must be "normalize" or "standardize".
        kwargs:
            Additional keyword arguments to pass to the preprocessing function.

    Returns:

    output: pd.DataFrame.
            The preprocessed DataFrame.
    parameters: dict, optional
            If return_params is True, the function returns a tuple containing the preprocessed DataFrame and the preprocessing parameters.
    """
    # Define a dictionary mapping method names to functions
    options = {"normalize": normalize, "standardize": standardize}
    # Select the appropriate function based on the method name
    output = options[method](table.copy(), **kwargs)

    log.info("Numerical preprocessing done")

    return output


if __name__ == "__main__":
    pass
