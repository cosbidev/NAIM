import os
import re
import logging
import pandas as pd
from typing import Tuple

from .techniques import *
from CMC_utils import save_load
from CMC_utils.miscellaneous import do_really_nothing
from CMC_utils.paths import get_directories

log = logging.getLogger(__name__)


__all__ = ["set_cross_validation", "get_cross_validation"]


def set_cross_validation(info_for_cv: pd.DataFrame, path: str, test_params: dict, val_params: dict, force: bool = False, **kwargs) -> None:
    """
    Set up cross-validation folds for training and evaluating machine learning models.

    Parameters:
        info_for_cv: pd.DataFrame.
            Pandas DataFrame object containing info for each data point.
        path: str.
            Directory to save the created folds to.
        test_params: dict.
            Dictionary of additional parameters for the test cross-validation method.
        val_params: dict.
            Dictionary of additional parameters for the validation cross-validation method.
        force: bool, default=False.
            Boolean flag indicating whether to force the function to re-create the folds even if they already exist (defaults to False).
        kwargs:
            Additional keyword arguments to pass to the save_table function when saving the folds.

    Returns:

    None
    """
    # Create DataFrame containing information about the cross-validation methods and their parameters
    cv_info = pd.DataFrame(
        {"test": pd.Series(test_params), "val": pd.Series(val_params)}).T.rename_axis(
        "set")

    folds_info_path = os.path.join(path, "folds_info.csv")
    if os.path.exists(folds_info_path):
        old_folds_info = save_load.load_table(folds_info_path, index_col=0)

        if not force and old_folds_info.equals(cv_info):
            return
        else:
            save_load.delete_directory(path)

    # Dictionary mapping method names to the corresponding functions that implement them
    options = {"holdout": holdout, "kfold": kfold, "stratifiedkfold": stratifiedkfold, "loo": leaveoneout,
               "bootstrap": bootstrap, "predefined": predefined}

    info_for_cv = info_for_cv.copy().reset_index()
    # Choose test cross-validation method and call it with the IDs and labels data and test_params parameters
    test_cv = options[test_params["method"]](info_for_cv.ID.values, info_for_cv.target.values, **test_params)

    for fold, train_val_index, test_index in test_cv:
        # Create DataFrame containing information about the test set
        test_info = pd.DataFrame(
            {"idx": test_index, "ID": info_for_cv.ID[test_index].values, "target": info_for_cv.target.iloc[test_index].values})

        # Split training and validation data using the validation cross-validation method
        train_val_IDs, train_val_info = info_for_cv.ID[train_val_index], info_for_cv.target.iloc[train_val_index].reset_index().rename({"index": "original_index"}, axis=1)
        val_cv = options[val_params["method"]](train_val_IDs.values, train_val_info.target.values, **val_params)

        for val_fold, train_index, val_index in val_cv:
            train_original_index = train_val_info.original_index.iloc[train_index]
            val_original_index = train_val_info.original_index.iloc[val_index]

            # Create DataFrame containing information about the train set
            train_info = pd.DataFrame({"idx": train_original_index.values, "ID": info_for_cv.ID.iloc[train_original_index].values,
                                       "target": info_for_cv.target.iloc[train_original_index].values})

            # Create DataFrame containing information about the validation set
            val_info = pd.DataFrame({"idx": val_original_index.values, "ID": info_for_cv.ID.iloc[val_original_index].values,
                                     "target": info_for_cv.target.iloc[val_original_index].values})

            fold_path = os.path.join( path, f"{fold}_{val_fold}")
            os.makedirs(fold_path)

            save_load.save_table(train_info, "train.csv", fold_path, mode='w', **kwargs)
            save_load.save_table(val_info, "val.csv", fold_path, mode='w', **kwargs)
            save_load.save_table(test_info, "test.csv", fold_path, mode='w', **kwargs)

    save_load.save_table(cv_info, "folds_info.csv", path, mode='w', index=True, **kwargs)
    log.info("Cross-validation folds ready")


def get_cross_validation(path: str, *sets: str, print_fold: bool = True) -> Tuple[int, int, pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    """
    Iterate over the cross-validation folds created by the set_cross_validation function.

    Parameters:
        path: str.
            Directory where the folds are stored.
        print_fold: bool.
            Whether to print fold number or not.

    Yields:

    Tuple containing the test fold number, validation fold number, and Pandas DataFrames containing the indices,
        IDs, and labels for the training, validation, and test sets.
    """
    # Get file paths for the folds
    folds_paths = get_directories(path)

    # Sort file paths by validation fold number and then by test fold number
    folds_paths = sorted(folds_paths, key=lambda file_path: int(re.findall(r"\d+_(\d+)\Z", file_path)[0]))
    folds_paths = sorted(folds_paths, key=lambda file_path: int(re.findall(r"(\d+)_\d+\Z", file_path)[0]))

    # Extract test and validation fold numbers from file names and store them in a DataFrame
    folds = pd.DataFrame([re.findall(r"(\d+)_(\d+)\Z", file_path)[0] for file_path in folds_paths], columns=["test", "val"])

    # Calculate the maximum validation fold number
    max_val_fold = folds.val.max()
    print_options = {True: log.info, False: do_really_nothing}
    for idx, fold_path in enumerate(folds_paths):
        # Get test and validation fold numbers for current fold
        fold, val_fold = folds.iloc[idx]

        # Determine whether the current validation fold is the last one for the current test fold
        last_val_fold = val_fold == max_val_fold  # and max_val_fold != 0

        # Load information about the current fold from the file
        fold_info = [save_load.load_table(os.path.join(fold_path, f"{fset}.csv")) for fset in sets]

        print_options[print_fold](f"Fold {fold}-{val_fold}: started")
        # Yield a tuple containing the test fold number, validation fold number, and DataFrames for the training, validation, and test sets
        yield int(fold), int(val_fold), *fold_info, last_val_fold


if __name__ == "__main__":
    pass
