import os
import logging
import pandas as pd
from typing import Tuple, Union
from .tables import load_table

__all__ = ["load_tabular_dataset"]

log = logging.getLogger(__name__)


def load_tabular_dataset(path: str, columns: dict, **kwargs ) -> Tuple[ pd.DataFrame, pd.DataFrame, pd.Series ]:
    """
    Loads a tabular dataset.

    Parameters:
        path: str
            Path to the folder containing the dataset.
        columns: dict
            Dictionary containing the features types.

    Returns:

    data: pd.DataFrame.
        Pandas DataFrame containing the features.
    labels: pd.DataFrame.
        Pandas DataFrame containing the features.
    features_types: pd.Series.
        Pandas Series containing the features types.
    """

    load_params = kwargs.get("pandas_load_kwargs", None)
    load_params = {} if load_params is None else load_params

    if load_params.get("header", None) is None and load_params.get("names", None) is None:
        columns_names = list(columns.keys())
    else:
        columns_names = load_params.get("names", None)

    data = load_table( path, names=columns_names, **load_params )

    target_name = [column_name for column_name, column_type in columns.items() if column_type == "target" ]
    id_col = [column_name for column_name, column_type in columns.items() if column_type == "id"]

    columns = {column_name: columns[column_name] for column_name in data.columns}

    if "id" in list(columns.values()) and load_params.get("index_col", None) is None:
        data = data.set_index(id_col)
        for col in id_col:
            del columns[col]

    data = data.rename_axis(["ID"] + id_col[1:], axis=0)

    data = data.dropna(subset=target_name, how="any")

    labels = data.loc[:, target_name].reset_index()
    data = data.drop(target_name, axis=1).reset_index()

    label_col = target_name if isinstance(target_name, str) else target_name[0]
    labels = labels.rename({label_col: "label"}, axis=1)

    data["ID"] = data.ID.astype(str)
    labels["ID"] = labels.ID.astype(str)

    data.set_index("ID", inplace=True)
    labels.set_index("ID", inplace=True)

    features_types = pd.Series(columns, name="dtype").drop(target_name)
    log.info("Data loaded")

    return data, labels, features_types


if __name__ == "__main__":
    pass
