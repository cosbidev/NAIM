import logging
import numpy as np
import pandas as pd
from typing import Union, Tuple

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer, MissingIndicator

log = logging.getLogger(__name__)

__all__ = ["impute_missing"]


def get_nan_mask(table: pd.DataFrame, missing_values: int = np.nan, features: str = "all", sparse: bool = "auto", error_on_new: bool = True ) -> pd.DataFrame:
    """
    This function returns a mask of missing values in the input table.

    Parameters:
        missing_values: int, float, str, np.nan or None, default=np.nan
            The placeholder for the missing values. All occurrences of missing_values will be imputed. For pandas’ dataframes with nullable integer dtypes with missing values, missing_values should be set to np.nan, since pd.NA will be converted to np.nan.
        features: {‘missing-only’, ‘all’}, default=’all’
            Whether the imputer mask should represent all or a subset of features.
                - If 'missing-only' (default), the imputer mask will only represent features containing missing values during fit time.
                - If 'all', the imputer mask will represent all features.
        sparse: bool or ‘auto’, default=’auto’
            Whether the imputer mask format should be sparse or dense.
                - If 'auto' (default), the imputer mask will be of same type as input.
                - If True, the imputer mask will be a sparse matrix.
                - If False, the imputer mask will be a numpy array.
        error_on_new: bool, default=True
            If True, transform will raise an error when there are features with missing values that have no missing values in fit. This is applicable only when features='missing-only'.

    Returns:

    mask: pd.DataFrame.
        The mask of missing values in the input table.
    """
    mask_generator = MissingIndicator(missing_values=missing_values, features=features, sparse=sparse, error_on_new=error_on_new)
    mask_generator = mask_generator.fit(table)
    mask = pd.DataFrame(mask_generator.transform(table), columns=table.columns)
    return mask


def impute_missing(data: pd.DataFrame, features_info: Union[dict, pd.Series], method: str, return_mask: bool = False,
                   return_imputer: bool = False, imputer: object = None, concat_mask: bool = False, **kwargs) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, object], Tuple[pd.DataFrame, pd.DataFrame, object]]:
    """
    Impute missing values in a DataFrame using a specified imputation method.

    Parameters:
        data: pd.DataFrame.
            The DataFrame to impute missing values in.
        features_info: pd.Series.
            A Series indicating the data type of each column in `data`.
        method: str.
            The imputation method to use. Can be one of 'simple', 'iterative', 'knn', or 'no_imputation'.
                - 'simple': use scikit-learn's SimpleImputer.
                - 'iterative': use scikit-learn's IterativeImputer.
                - 'knn': use scikit-learn's KNNImputer.
                - 'no_imputation': don't impute missing values, just return a mask indicating which values are missing.
        return_mask: bool, default=False.
            Whether to return a mask indicating which values in `data` are missing. Default is False.
        return_imputer: bool, default=False.
            Whether to return the imputer object used to impute missing values. Default is False.
        imputer: object.
            The imputer object to use. If not provided, one will be created and fit to `data`.
        concat_mask: bool, default=False.
            Whether to concatenate the mask to the data as missing indicator or not.
        **kwargs:
            additional keyword arguments to pass to the imputer object's fit method.

    Returns:

    table: pd.DataFrame
        if both `return_mask` and `return_imputer` are False, returns the imputed DataFrame.
    mask: pd.DataFrame
        if `return_mask` is True, returns a tuple containing also the missing value mask.
    imputer: object
        if `return_imputer` is True, returns a tuple containing also and the imputer object.
    """

    # get columms names
    dataset_columns = data.columns

    # define imputer options
    options = dict(simple=SimpleImputer, iterative=IterativeImputer, knn=KNNImputer, no_imputation=MissingIndicator)

    if imputer is None:
        imputer = options[method](**kwargs).fit(data)

    # fill nan values
    data_filled = data.copy()
    if method != "no_imputation":
        data_filled = imputer.transform(data_filled)
        data_filled = np.where(np.isfinite(data_filled.astype("float32")), data_filled, np.ma.masked_invalid(data_filled.astype("float32")).mean(axis=0))  ##
        data_filled = pd.DataFrame(data_filled, index=data.index, columns=dataset_columns)

        categorical_columns = features_info["features_types"][features_info["features_types"] == "category"].index.tolist()
        for cat_column in categorical_columns:
            # get the list of columns (1-hot) linked to the current column
            relative_columns = dataset_columns[dataset_columns.str.startswith(f"{cat_column}")].to_list()

            # if there is just one consider it as a discrete variable -> round the numbers
            if len(relative_columns) == 1:
                data_filled[relative_columns] = data_filled[relative_columns].round(0)
                data_filled[relative_columns] = data_filled[relative_columns].clip(0, len(features_info["categorical_unique_values"].loc[relative_columns]))

            # otherwise set to 1 just the argmax of the different columns
            else:
                new_values = pd.DataFrame(0, index=data_filled.index, columns=relative_columns)

                new_values = new_values.stack()
                new_values.loc[zip(data_filled.index.tolist(), data_filled[relative_columns].idxmax(axis="columns").to_list())] = 1
                new_values = new_values.unstack()

                data_filled[relative_columns] = new_values

    # get nan mask
    mask = get_nan_mask( data )

    if concat_mask:
        data_filled = pd.concat( [data_filled, mask], axis=1 )

    output_options = {(False, False): data_filled, (True, False): (data_filled, mask),
                      (False, True): (data_filled, imputer), (True, True): (data_filled, mask, imputer)}

    log.info("Imputation done")

    return output_options[(return_mask, return_imputer)]


if __name__ == "__main__":
    pass
