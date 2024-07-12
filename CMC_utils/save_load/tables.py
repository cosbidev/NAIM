import os
import ast
import logging
import pandas as pd
from typing import Union

from CMC_utils.miscellaneous import filter_kwargs
from CMC_utils.paths import add_extension, get_extension

__all__ = ["load_table", "save_table", "load_params_table"]

log = logging.getLogger(__name__)

## load

def _load_csv_table(path: str, **kwargs) -> pd.DataFrame:
    """
    This function loads a csv or txt file.

    Parameters:
        path: string.
            Path to the csv file to load.
        kwargs:
            Keyword arguments to pass on to the read_csv function.

    Returns:

    table: pd.DataFrame.
        Table loaded.

    """
    filtered_kwargs = filter_kwargs(pd.read_csv, **kwargs)
    table = pd.read_csv(path, **filtered_kwargs)
    return table


def _load_excel_table(path: str, **kwargs) -> pd.DataFrame:
    """
    This function loads a xlsx/xls file.

    Parameters:
        path: string.
            Path to the csv file to load.
        kwargs:
            Keyword arguments to pass on to the read_excel function.

    Returns:

    table: pd.DataFrame.
        Table loaded.

    """
    filtered_kwargs = filter_kwargs(pd.read_excel, **kwargs)
    table = pd.read_excel(path, **filtered_kwargs)
    return table


def load_table(path: str, **kwargs) -> pd.DataFrame:
    """
    This function load a table, independently of the extension it has.

    Parameters:
        path: string.
            Path to the table to load.
        kwargs:
            Keyword arguments to pass on the read function.

    Returns:

    table: pd.DataFrame.
        Table loaded.

    """
    extension = get_extension(path).lower()

    options = {"csv": _load_csv_table,
               "txt": _load_csv_table,
               "data": _load_csv_table,
               "test": _load_csv_table,
               "arff": _load_csv_table,
               "xlsx": _load_excel_table,
               "xls": _load_excel_table}

    table = options[extension](path, **kwargs)

    return table


def load_params_table(path: str, **kwargs) -> pd.DataFrame:
    """
    This function load a table, independently of the extension it has.

    Parameters:
        path: string.
            Path to the table to load.
        kwargs:
            Keyword arguments to pass on the read function.

    Returns:

    table: pd.DataFrame.
        Table loaded.

    """

    def parse_literal_list(literal_str):
        try:
            return ast.literal_eval(literal_str)
        except (SyntaxError, ValueError):
            return literal_str

    data = load_table(path, **kwargs)
    data = data.applymap(parse_literal_list)
    return data

## save

def _save_table_as_csv(table: Union[ pd.DataFrame, pd.Series ], path: str, mode="w", index=False, engine="openpyxl", **kwargs) -> None:
    """
    This function save a csv/txt file.

    Parameters:
        table: pd.DataFrame | pd.Series
            Table to be saved.
        path: string.
            Path to save the table.
        mode: string, default = "w".
            Python write mode. The available write modes are [ "w", "a" ].
        index: bool, default = False.
            Write row names (index).
        engine: string.
            Parameter added for compatibility.
        kwargs:
            Keyword arguments to pass on to the 'to_csv' function.

    Returns:

    None

    """
    filtered_kwargs = filter_kwargs(pd.DataFrame.to_csv, **kwargs)
    table.to_csv(path, mode=mode, index=index, **filtered_kwargs)


def _save_table_as_excel(table: Union[ pd.DataFrame, pd.Series ], path: str, mode="w", index=False, engine="openpyxl", **kwargs) -> None:
    """
    This function save a xlsx/xls file.

    Parameters:
        table: pd.DataFrame | pd.Series.
            Table to be saved.
        path: string.
            Path to save the table.
        mode: string, default "w"
            Python write mode. The available write modes are [ "w", "a" ].
        index: bool, default = False.
            Write row names (index).
        engine: string, default = "openpyxl".
            Write engine to use, ‘openpyxl’ or ‘xlsxwriter’.
        kwargs:
            Keyword arguments to pass on to the 'to_csv' function.

    Returns:

    None

    """
    writer = pd.ExcelWriter(path, mode=mode, engine=engine)

    filtered_kwargs = filter_kwargs(pd.DataFrame.to_excel, **kwargs)
    table.to_excel(writer, index=index, **filtered_kwargs)

    # writer.save()
    writer.close()


def save_table(table: Union[ pd.DataFrame, pd.Series ], table_name: str, directory_path: str, extension: str = None, mode: str = 'w', **kwargs) -> None:
    """
    This function saves a table, independently of the extension it has.

    Parameters:
        table: pd.DataFrame | pd.Series.
            Table to be saved.
        table_name: string.
            Name of the file to save.
        directory_path: string.
            Path to the folder where to save the table.
        extension: sting.
            Extension to use to save.
        mode: string, default "w".
            Python write mode. The available write modes are [ "w", "a" ].
        kwargs:
            Keyword arguments to pass on to the save function.

    Returns:

    None

    """
    if extension is not None:
        table_name = add_extension(table_name, extension)
    else:
        extension = get_extension(table_name).lower()

    assert extension is not None, "Table extension not provided"

    path = os.path.join(directory_path, table_name)

    file_exists = os.path.exists(path)
    mode = mode * file_exists + 'w' * (not file_exists)
    header = kwargs.pop("header", False if mode == "a" else True)

    engine = kwargs.get("engine", "xlrd"*(extension == "xls") + "openpyxl"*(extension != "xls"))

    options = {"csv": _save_table_as_csv,
               "txt": _save_table_as_csv,
               "data": _save_table_as_csv,
               "test": _save_table_as_csv,
               "arff": _save_table_as_csv,
               "xls": _save_table_as_excel,
               "xlsx": _save_table_as_excel}

    options[extension](table, path, mode, engine=engine, header=header, **kwargs)


if __name__ == "__main__":
    pass
