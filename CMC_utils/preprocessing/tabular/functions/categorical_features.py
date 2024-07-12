import logging
import numpy as np
import pandas as pd
from typing import Union, Tuple

from .miscellaneous import get_unique_values

log = logging.getLogger(__name__)

__all__ = ["categorical_preprocessing"]

def one_hot_encode(table: pd.DataFrame, unique_values: Union[ dict, pd.Series ] = None,
                   nan_as_category: bool = False, fill_value: int = None,
                   return_categories: bool = False, **kwargs) -> Union[ pd.DataFrame, Tuple[ pd.DataFrame, dict ] ]:
    """
    One-hot encode a Pandas DataFrame.

    Parameters:
        table: pd.DataFrame.
            A Pandas DataFrame.
        unique_values: dict | pd.Series, optional.
            A dictionary or Pandas Series containing the unique values for each column in the DataFrame.
            If not provided, this is obtained by calling the `get_unique_values` function on `table`.
        nan_as_category: bool, default = False.
            A boolean indicating whether to include a separate column for `NaN` values in the resulting
            one-hot encoded DataFrame.
        fill_value: int or None, optional.
            The value to use for filling missing values if `nan_as_category` is `False`, if None NaN values
            are preserved.
        return_categories: bool, default = False.
            A boolean indicating whether to return the unique_values or not.
        kwargs:
            Additional keyword arguments to pass on to the 'get_unique_values' function.

    Returns:

    table_1hot: pd.DataFrame.
        A one-hot encoded version of the input DataFrame, where each categorical column is replaced by
        multiple binary columns, one for each category. If `nan_column` is `True`, it includes a separate
        column for `NaN` values in the resulting DataFrame. If `fill_missing` is `True`, it fills missing
        values with the specified `fill_value`.
    unique_values: dict, optional.
        If return_params is True, returns a tuple containing the one-hot encoded version of the input
        DataFrame and the dict containing the unique values for each column in the DataFrame.
    """

    # If `unique_values` is not provided, get it by calling the `get_unique_values` function on `table`
    if unique_values is None:
        unique_values = get_unique_values(table, nan_as_category=nan_as_category, **kwargs)

    if isinstance(unique_values, pd.Series):
        unique_values = unique_values.to_dict()

    # Make a copy of the input DataFrame
    table_1hot = table.copy()

    columns_to_drop = []
    # Iterate over each column in the DataFrame
    for column_name, categories in unique_values.items():
        if not nan_as_category and any(pd.isna(categories)):
            categories = list(filter(lambda x: not pd.isna(x), categories))  # categories[ np.logical_not(np.isnan(categories)) ]

        if len(categories) <= 2:
            for cat_idx, category in enumerate(categories):
                # Set the values in the new column to the category index if the corresponding value in the original column is equal to `category`, and 0 otherwise
                table_1hot.loc[table_1hot.loc[:, column_name] == category, column_name] = cat_idx
        else:
            columns_to_drop.append( column_name )
            # Iterate over each category in the current column
            for category in categories:
                # Create a new column with the name `column_name_category`
                new_column_name = f"{column_name}_{category}"

                new_column = pd.DataFrame({new_column_name: (table_1hot[column_name].values == category).astype(int)}, index=table_1hot.index)
                table_1hot = pd.concat( [table_1hot, new_column], axis=1 )
                table_1hot[new_column_name] = np.where(table_1hot[column_name].isna(), np.nan, table_1hot[new_column_name])

        # If `nan_as_category` is True and there are `NaN` values in the current column
        if nan_as_category and table_1hot.loc[:, column_name].isna().any(axis=None):
            # Create a new column with the name `column_name_nan`
            table_1hot[f"{column_name}_nan"] = table_1hot[column_name].isna().astype(int)

    # Drop the original columns from the DataFrame
    table_1hot = table_1hot.drop(columns_to_drop, axis=1)

    if nan_as_category:
        fill_value = 0
    if fill_value not in ( None, np.nan ):
        # Fill missing values (`NaN`) in the DataFrame with the specified `fill_value`
        table_1hot = table_1hot.fillna( fill_value ).astype(int)

    parameters = dict(method="one_hot_encode", unique_values=unique_values, nan_as_category=nan_as_category, fill_value=fill_value )
    output_options = {False: table_1hot, True: (table_1hot, parameters)}
    return output_options[return_categories]


def categorical_encode(table: pd.DataFrame, unique_values: Union[dict, pd.Series] = None, nan_as_category: bool = False,
                       fill_value: int = None, return_categories: bool = False, **kwargs) -> Union[ pd.DataFrame, Tuple[ pd.DataFrame, dict]]:
    """
    Categorical encodes a Pandas DataFrame.

    Parameters:
        table: pd.DataFrame.
            A Pandas DataFrame
        unique_values: dict | pd.Series, optional.
            A Pandas Series containing the unique values for each column in the DataFrame. If not provided, this is obtained by calling the `get_unique_values` function on `table`.
        nan_as_category: bool, default = False.
            A boolean indicating whether to include `NaN` values as a separate category in the resulting categorical encoded DataFrame. Default is False.
        fill_value: int, optional.
            The value to use for filling missing values if `nan_as_category` is `False`, if None 'NaN' values are preserved.
        return_categories: bool, default = False.
            A boolean indicating whether to return the unique_values or not.
        kwargs:
            Additional keyword arguments to pass on to the 'get_unique_values' function.

    Returns:

    table_encoded: pd.DataFrame.
        A categorical encoded version of the input DataFrame, where each categorical column is encoded with
        successive numbers, one for each category. If `nan_column` is `True`, it includes `NaN` values as a
        separate category in the resulting DataFrame. If `fill_missing` is `True`, it fills missing values
        with the specified `fill_value`.
    unique_values: dict, optional.
        If return_params is True, returns a tuple containing the categorical encoded version of the input
        DataFrame and the dict containing the unique values for each column in the DataFrame.
    """
    # If `unique_values` is not provided, get it by calling the `get_unique_values` function on `table`
    if unique_values is None:
        unique_values = get_unique_values(table, nan_as_category=nan_as_category, **kwargs)

    if isinstance(unique_values, pd.Series):
        unique_values = unique_values.to_dict()

    # Make a copy of the input DataFrame
    table_encoded = table.copy()

    # Iterate over each column in the DataFrame
    for column_name, categories in unique_values.items():

        if not nan_as_category and any(pd.isna(categories)):
            categories = list(filter(lambda x: not pd.isna(x), categories))

        # Iterate over each category in the current column
        for cat_idx, category in enumerate(categories):
            # Set the values in the new column to the category index if the corresponding value in the original column is equal to `category`, and 0 otherwise
            table_encoded.loc[ table[column_name] == category, column_name ] = cat_idx

        if nan_as_category:
            fill_value = (~pd.isna(categories)).sum()

        if fill_value is not None:
            # Set the values in the new column to `NaN` if the corresponding value in the original column is `NaN`
            table_encoded[column_name].where(table[column_name].notna(), fill_value, inplace=True)

    parameters = dict(method= "categorical_encode", unique_values=unique_values, nan_as_category=nan_as_category, fill_value=fill_value)
    output_options = {False: table_encoded, True: (table_encoded, parameters)}
    return output_options[return_categories]


def categorical_preprocessing(table: pd.DataFrame, method: str = "one_hot_encode", **kwargs) -> Union[ pd.DataFrame, Tuple[ pd.DataFrame, dict]]:
    """
    Applies categorical preprocessing to a Pandas DataFrame.

    Parameters:
    table: pd.DataFrame.
        The DataFrame to preprocess.
    method: str, default = 'one_hot_encode'.
        The method to use for preprocessing. Must be "one_hot_encode" or "categorical_encode". Default is "one_hot_encode".
    kwargs:
        Additional keyword arguments to pass to the preprocessing function.

    Returns:

    output: pd.DataFrame.
        The preprocessed DataFrame.
    parameters: dict, optional.
        If return_params is True, returns a tuple containing the preprocessed DataFrame and the preprocessing parameters.
    """
    # Define a dictionary mapping method names to functions
    options = {"one_hot_encode": one_hot_encode, "categorical_encode": categorical_encode }
    # Select the appropriate function based on the method name
    output = options[method](table, **kwargs)

    log.info("Categorical preprocessing done")

    return output


if __name__ == "__main__":
    pass
