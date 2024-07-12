import logging
import numpy as np
import pandas as pd
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=False)

from typing import Union, Tuple

__all__ = ["generate_missing"]

log = logging.getLogger(__name__)

def MCAR_vector_mask(data: Union[pd.Series, np.array], missing_fraction: float, index_direction: int) -> np.array:
    """
    Generate a mask for a vector of data, where the missing values are randomly distributed according to a missing fraction
    Parameters
    ----------
    data : Union[pd.Series, np.array]
    missing_fraction : float
    index_direction : int

    Returns
    -------
    np.array
    """
    data_length = len(data)
    values_to_remove = int(np.floor(missing_fraction * data_length))
    values_already_missing = data.isna().values
    values_to_remove = np.maximum( values_to_remove - values_already_missing.sum(), 0 )

    missing_mask = np.hstack([np.ones(values_to_remove), np.zeros(data_length - values_to_remove)])  # ones to be removed, zeros to keep

    np.random.shuffle(missing_mask)

    missing_mask = missing_mask + values_already_missing
    missing_masked = np.where(missing_mask == 2)[0]
    not_masked = np.where(missing_mask == 0)[0]

    not_masked_to_mask = np.random.choice(not_masked, size=len(missing_masked), replace=False)
    missing_mask[not_masked_to_mask] = 1
    missing_mask[missing_masked] = 1

    missing_mask = np.expand_dims( missing_mask.astype(bool), axis=index_direction )

    return missing_mask


def mask_correction(mask: np.array, axis_to_check: int, data: np.array) -> np.array:
    """
    Correct the mask to avoid completely missing rows or columns
    Parameters
    ----------
    mask : np.array
    axis_to_check : int
    data : np.array

    Returns
    -------
    np.array
    """
    completely_missing_map = mask.all(axis=axis_to_check)

    if completely_missing_map.any():

        completely_missing_idxs = completely_missing_map.nonzero()[0]

        # selectable_second_axis_idxs = ((~mask).sum(axis=int(not axis_to_check)) > 1).nonzero()[0]

        for missing_idx in completely_missing_idxs:
            selectable_second_axis_idxs = (np.invert(mask).sum(axis=int(not axis_to_check)) > 1).nonzero()[0]

            element_corrected = False
            while not element_corrected and len(selectable_second_axis_idxs) > 0:
                selected_second_axis_idx = np.random.choice( selectable_second_axis_idxs )

                axis_second_condition_options = {0: (selected_second_axis_idx, np.arange(mask.shape[1])), 1: (np.arange(mask.shape[0]), selected_second_axis_idx)}
                if axis_to_check == 0:
                    vector = data.iloc[:, missing_idx]
                else:
                    vector = data.iloc[missing_idx, :]

                change_index = vector.isna()[selected_second_axis_idx]
                if change_index:
                    selectable_second_axis_idxs = np.delete(selectable_second_axis_idxs, np.where(selectable_second_axis_idxs == selected_second_axis_idx))
                    continue

                first_condition_selectable_map = np.invert(mask).sum( axis=axis_to_check ) > 1
                second_condition_selectable_map = np.invert(mask)[axis_second_condition_options[axis_to_check]]
                third_condition_selectable_map = np.invert(data.iloc[axis_second_condition_options[axis_to_check]].isna())
                selectable_main_axis_idxs = np.vstack( [first_condition_selectable_map, second_condition_selectable_map, third_condition_selectable_map] ).all(axis=0).nonzero()[0]

                if len(selectable_main_axis_idxs) == 0:
                    selectable_second_axis_idxs = np.delete(selectable_second_axis_idxs, np.where(selectable_second_axis_idxs == selected_second_axis_idx))
                else:
                    selected_main_axis_idx = np.random.choice( selectable_main_axis_idxs )

                    to_be_deleted_options = {0: (selected_second_axis_idx, selected_main_axis_idx), 1: (selected_main_axis_idx, selected_second_axis_idx)}
                    to_be_corrected_options = {0: (selected_second_axis_idx, missing_idx), 1: (missing_idx, selected_second_axis_idx)}

                    mask[to_be_deleted_options[axis_to_check]] = True
                    mask[to_be_corrected_options[axis_to_check]] = False

                    element_corrected = True
            assert element_corrected, "There are too few not-missing values to correct the mask"

    return mask


def MCAR_feature(data: pd.DataFrame, missing_fraction: float, **_) -> Tuple[pd.DataFrame, np.array]:
    """
    Generate a mask for a vector of data, where the missing values are randomly distributed according to a missing fraction
    Parameters
    ----------
    data : Union[pd.Series, np.array]
    missing_fraction : float

    Returns
    -------
    np.array
    """
    masked_data = data.copy()

    generation_direction = 1

    missing_mask = np.array([], dtype=bool).reshape(masked_data.shape[0], 0)

    for feature_idx in range(masked_data.shape[generation_direction]):
        missing_vector_mask = MCAR_vector_mask(masked_data.iloc[:, feature_idx], missing_fraction, index_direction=generation_direction)
        missing_mask = np.hstack([missing_mask, missing_vector_mask])

    missing_mask = mask_correction(missing_mask, generation_direction, masked_data)

    masked_data = masked_data.mask(missing_mask, np.nan)

    final_missing_percentage = np.mean( np.round((masked_data.isna().sum(axis=0) / masked_data.shape[0]) * 100) )
    log.info(f"{final_missing_percentage}% of column data is missing in average")

    return masked_data, final_missing_percentage


def MCAR_sample(data: pd.DataFrame, missing_fraction: float, **_) -> Tuple[pd.DataFrame, np.array]:
    """
    Generate a mask for a vector of data, where the missing values are randomly distributed according to a missing fraction
    Parameters
    ----------
    data : Union[pd.Series, np.array]
    missing_fraction : float

    Returns
    -------
    np.array
    """
    masked_data = data.copy()

    generation_direction = 0

    missing_mask = np.array([], dtype=bool).reshape(0, masked_data.shape[1])

    for sample_idx in range(masked_data.shape[generation_direction]):
        missing_vector_mask = MCAR_vector_mask(masked_data.iloc[sample_idx, :], missing_fraction, index_direction=generation_direction)
        missing_mask = np.vstack([missing_mask, missing_vector_mask])

    missing_mask = mask_correction(missing_mask, generation_direction, masked_data)

    masked_data = masked_data.mask(missing_mask, np.nan)

    final_missing_percentage = np.mean( np.round((masked_data.isna().sum(axis=1) / masked_data.shape[1]) * 100) )
    log.info(f"{final_missing_percentage}% of row data is missing in average")

    return masked_data, final_missing_percentage


def MCAR_global(data: pd.DataFrame, missing_fraction: float, **_) -> Tuple[pd.DataFrame, np.array]:
    """
    Generate a mask for a vector of data, where the missing values are randomly distributed according to a missing fraction
    Parameters
    ----------
    data : Union[pd.Series, np.array]
    missing_fraction : float

    Returns
    -------
    np.array
    """
    masked_data = data.copy()

    missing_mask = MCAR_vector_mask(pd.Series(masked_data.to_numpy().flatten()), missing_fraction, index_direction=0)
    missing_mask = missing_mask.reshape(masked_data.shape)

    missing_mask = mask_correction(missing_mask, 0, masked_data)
    missing_mask = mask_correction(missing_mask, 1, masked_data)

    masked_data = masked_data.mask(missing_mask, np.nan)

    final_missing_percentage = np.round((masked_data.isna().sum().sum() / (masked_data.shape[0] * masked_data.shape[1]))*100)
    log.info( f"{final_missing_percentage}% of data is missing" )

    return masked_data, final_missing_percentage


def no_generation(data: pd.DataFrame, **_) -> Tuple[pd.DataFrame, np.array]:
    final_missing_percentage = np.round((data.isna().sum().sum() / (data.shape[0] * data.shape[1])) * 100)
    log.info(f"{final_missing_percentage}% of data is missing")
    return data, final_missing_percentage


def generate_missing( data: pd.DataFrame, method: str, missing_fraction: float, copy: bool = True, **kwargs ) -> Tuple[pd.DataFrame, np.array]:
    """
    Generate a mask for a vector of data, where the missing values are randomly distributed according to a missing fraction
    Parameters
    ----------
    data : Union[pd.Series, np.array]
    method : str
    missing_fraction : float
    copy : bool
    kwargs : dict

    Returns
    -------
    np.array
    """
    if copy:
        data = data.copy()

    options = dict( MCAR_sample = MCAR_sample, MCAR_feature = MCAR_feature, MCAR_global = MCAR_global, no_generation=no_generation )

    params = { "missing_fraction": missing_fraction, **kwargs }  # "return_first": True,

    IDs = data.index.to_list()

    masked_data = data.reset_index(drop=True)  # .drop("ID", axis=1)

    masked_data, final_missing_percentage = options[ method ]( masked_data, **params )
    assert not masked_data.isna().all(axis=0).any() and not masked_data.isna().all(axis=1).any()

    masked_data = masked_data.assign(ID=IDs).set_index("ID")

    return masked_data, final_missing_percentage


if __name__ == "__main__":
    pass
