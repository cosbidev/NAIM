import logging
import numpy as np
import pandas as pd

from omegaconf import DictConfig
from typing import Tuple, Union

log = logging.getLogger(__name__)

__all__ = ["get_preprocessing_params", "get_columns_lists_by_type", "get_unique_values"]


def get_columns_lists_by_type(features_info: pd.Series) -> Tuple[ dict, dict ]:
    """
    Gets the list of the columns by type.

    Parameters:
        features_info: pd.Series.
            Pandas Series containing the features types.

    Returns:

    numerical_columns: list
        List containing the numerical columns names.
    categorical_columns: list
        List containing the categorical columns names.

    """
    check_numerical = lambda feature_type: True if feature_type in ("int", "float") else False
    check_categorical = lambda feature_type: True if feature_type == "category" else False

    features_info = features_info.rename_axis("feature").rename("dtype").reset_index()
    numerical_columns = features_info.loc[features_info.dtype.map(check_numerical), "feature"].to_dict()
    categorical_columns = features_info.loc[features_info.dtype.map(check_categorical), "feature"].to_dict()

    return numerical_columns, categorical_columns


def get_unique_values(table: pd.DataFrame, nan_as_category: bool = False, categorical_columns: list = None) -> pd.Series:
    """
    Get unique values for each column in a Pandas DataFrame.

    Parameters:
        table: pd.DataFrame.
            A Pandas DataFrame.
        nan_as_category: bool, default = False.
            A boolean indicating whether to include `NaN` values as a category.
        categorical_columns: list, optional.
            A list of columns to apply the preprocessing to. If not provided, the preprocessing is applied to all columns.

    Returns:

    unique_values: pd.Series.
        Pandas Series containing the unique values for each categorical column in the DataFrame.
    """
    # If columns is not provided, apply the preprocessing to all columns
    if categorical_columns is None:
        categorical_columns = table.columns.tolist()

    unique_values = {}

    # Iterate over each column in the DataFrame
    for column_name in categorical_columns:
        # Get the values for the current column
        column = table[column_name]

        # Filter out `NaN` values if `nan_as_category` is False, otherwise keep all values
        column_map = {False: ~column.isna().values, True: np.ones( column.shape, dtype=bool ) }
        column = column[ column_map[nan_as_category] ]

        # Get the unique values for the current column and sort them
        values = column.sort_values().unique().tolist()
        if nan_as_category and np.nan not in values:
            unique_values[column_name] = [*values, np.nan]
        else:
            unique_values[column_name] = values

    # Create a Pandas Series with the unique values and return it
    unique_values = pd.Series(unique_values, name="categories", dtype=object)
    return unique_values


def get_preprocessing_params(data: pd.DataFrame, features_dtypes: pd.Series, preprocessing: Union[dict, DictConfig]) -> dict:
    """
    Get the preprocessing parameters.
    Parameters
    ----------
    data : pd.DataFrame
    features_dtypes : pd.Series
    preprocessing : Union[dict, DictConfig]

    Returns
    -------
    dict
        Preprocessing parameters.
    """
    numerical_columns, categorical_columns = get_columns_lists_by_type(features_dtypes)

    categorical_params = preprocessing.get( "categorical", None )
    if categorical_params is not None:
        categorical_unique_values = get_unique_values(data, nan_as_category=categorical_params["nan_as_category"], categorical_columns=list(categorical_columns.values()))
    else:
        categorical_unique_values = pd.Series(name="categories")

    output = dict( numerical_columns=numerical_columns, categorical_columns=categorical_columns, categorical_unique_values=categorical_unique_values, features_types=features_dtypes, **preprocessing)

    return output


if __name__ == "__main__":
    pass
