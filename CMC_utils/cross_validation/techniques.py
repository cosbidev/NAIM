import logging
import numpy as np
from typing import Tuple
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut, ShuffleSplit, StratifiedShuffleSplit

log = logging.getLogger(__name__)

__all__ = ["predefined", "holdout", "kfold", "stratifiedkfold", "leaveoneout", "bootstrap"]


def predefined(data: np.array, labels: np.array, idxs: list, **kwargs) -> Tuple[ int, np.array, np.array ]:
    """
    This function generates a predefined train/test split of the data and labels.
    Parameters
    ----------
    data : np.array
    labels : np.array
    idxs : list
    kwargs : dict

    Returns
    -------
    fold: int.
        Number of the fold.
    train_index: np.ndarray.
        The training set indices for that split.
    test_index: np.ndarray.
        The testing set indices for that split.
    """
    idxs = [idx if idx > 0 else data.shape[0]+idx for idx in idxs]
    eval_idxs = np.arange(*idxs)
    train_idxs = np.delete(np.arange(data.shape[0]), eval_idxs)
    for fold, (train_index, test_index) in enumerate([(train_idxs, eval_idxs)]):
        yield fold, train_index, test_index


def holdout(data: np.array, labels: np.array, train_size: float = None,
             test_size: float = None, random_state: int = None,
             stratify: bool = False, **kwargs) -> Tuple[ int, np.array, np.array ]:
    """
    Function to split data into train and test sets using holdout method.

    Parameters
    ----------
    data: np.array.
        Array of data to be split.
    labels: np.array.
        Array of labels corresponding to data.
    train_size: float, optional.
        Proportion of data to be used for training.
    test_size: float, optional.
        Proportion of data to be used for testing.
    random_state: int, optional.
        Seed for the random number generator.
    stratify: bool, optional.
        A boolean value indicating whether to stratify the split or not.

    Yields
    ----------
    fold: int
         Number of the fold
    train_index: np.ndarray.
        The training set indices for that split.
    test_index: np.ndarray.
        The testing set indices for that split.
    """
    if stratify:
        cv_function = StratifiedShuffleSplit
        # labels = stratify
    else:
        cv_function = ShuffleSplit

    cv = cv_function(n_splits=1, train_size=train_size, test_size=test_size, random_state=random_state)

    # train_index, test_index = next( cv.split( data, labels ) )
    # return 0, train_index, test_index
    for fold, (train_index, test_index) in enumerate(cv.split(data, labels)):
        yield fold, train_index, test_index


def kfold(data: np.array, labels: np.array, n_splits: int = 5, shuffle: bool = False,
           random_state: int = None, **kwargs) -> Tuple[ int, np.array, np.array ]:
    """
    Function to split data into train and test sets using k-fold method.

    Parameters
    ----------
    data: np.array.
        Array of data to be split.
    labels: np.array.
        Array of labels corresponding to data.
    n_splits: int, default = 5.
        Number of folds.
    shuffle: bool, default = False.
        Whether to shuffle the data before splitting.
    random_state: int, default = None.
        Seed for the random number generator.

    Yields
    ----------
    fold: int.
         Number of the fold.
    train_index: np.ndarray.
        The training set indices for that split.
    test_index: np.ndarray.
        The testing set indices for that split.
    """
    cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    for fold, (train_index, test_index) in enumerate(cv.split(data, labels)):
        yield fold, train_index, test_index


def stratifiedkfold(data: np.array, labels: np.array, n_splits: int = 5, shuffle: bool = False, random_state: int = None, **kwargs) -> Tuple[ int, np.array, np.array ]:
    """
    Function that performs stratified k-fold cross-validation.

    Parameters
    ----------
    data: np.array.
        Array of data points.
    labels: np.array.
        Array of labels corresponding to data points.
    n_splits: int, default = 5.
        Number of folds to use for cross-validation.
    shuffle: bool, default = False.
        Boolean flag indicating whether to shuffle the data before performing cross-validation.
    random_state: int, default = None.
        Random seed to use for shuffling data.

    Yields
    ----------
    fold: int.
         Number of the fold.
    train_index: np.ndarray.
        The training set indices for that split.
    test_index: np.ndarray.
        The testing set indices for that split.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    for fold, (train_index, test_index) in enumerate(cv.split(data, labels)):
        yield fold, train_index, test_index


def leaveoneout(data: np.array, labels: np.array, **kwargs) -> Tuple[ int, np.array, np.array ]:
    """
    Function that performs leave-one-out cross-validation.

    Parameters
    ----------
    data: np.array.
        Array of data points.
    labels: np.array.
        Array of labels corresponding to data points.

    Yields
    ----------
    fold: int.
         Number of the fold.
    train_index: np.ndarray.
        The training set indices for that split.
    test_index: np.ndarray.
        The testing set indices for that split.
    """
    cv = LeaveOneOut()

    for fold, (train_index, test_index) in enumerate(cv.split(data, labels)):
        yield fold, train_index, test_index


def bootstrap(data: np.array, labels: np.array, n_splits: int = 5, train_size: float = None, test_size: float = None, random_state: int = None, stratify: bool = False, **kwargs) -> Tuple[ int, np.array, np.array ]:
    """
    This function generates a bootstrapped train/test split of the data and labels.

    Parameters
    ----------
    data: np.array.
        The data to be split.
    labels: np.array.
        The labels for the data.
    n_splits: int.
        The number of splits to generate. Default is 5.
    train_size: int | float.
        The size of the training set. Either an integer or float (proportion of data).
    test_size: int | float.
        The size of the test set. Either an integer or float (proportion of data).
    random_state: int.
        The seed for the random number generator.
    stratify: bool, optional.
        A boolean value indicating whether to stratify the split or not.

    Yields
    ----------
    fold: int.
         Number of the fold.
    train_index: np.ndarray.
        The training set indices for that split.
    test_index: np.ndarray.
        The testing set indices for that split.
    """
    if stratify:
        cv_function = StratifiedShuffleSplit
        # labels = stratify
    else:
        cv_function = ShuffleSplit

    cv = cv_function(n_splits=n_splits, train_size=train_size, test_size=test_size, random_state=random_state)

    for fold, (train_index, test_index) in enumerate(cv.split(data, labels)):
        yield fold, train_index, test_index


if __name__ == "__main__":
    pass
