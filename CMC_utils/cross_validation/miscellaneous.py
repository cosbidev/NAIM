import pandas as pd
from typing import List, Tuple

__all__ = ["get_sets_with_idx", "get_sets_with_ID"]


def get_sets_with_idx(data: pd.DataFrame, *sets: pd.DataFrame, labels: pd.DataFrame = None) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    This function returns the data and labels of the sets passed as arguments.
    Parameters
    ----------
    data : pd.DataFrame
    sets : pd.DataFrame
    labels : pd.DataFrame

    Returns
    -------
    list
        List of data and labels of the sets passed as arguments.
    """
    sets_data = []
    for fset in sets:
        sets_data.append(data.iloc[fset.idx])
        if labels is not None:
            sets_data.append(labels.iloc[fset.idx])

    return sets_data


def get_sets_with_ID(data: pd.DataFrame, *sets: pd.DataFrame, labels: pd.DataFrame = None) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    This function returns the data and labels of the sets passed as arguments.
    Parameters
    ----------
    data : pd.DataFrame
    sets : pd.DataFrame
    labels : pd.DataFrame

    Returns
    -------
    list
        List of data and labels of the sets passed as arguments.
    """
    sets_data = []
    for fset in sets:
        sets_data.append(data.loc[fset.ID])
        if labels is not None:
            sets_data.append(labels.loc[fset.ID])

    return sets_data


if __name__ == "__main__":
    pass
