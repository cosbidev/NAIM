import numpy as np

__all__ = ["missing_augmentation"]


def missing_augmentation(sample: np.array) -> np.array:
    """
    Randomly mask a fraction of the sample
    Parameters
    ----------
    sample : np.array

    Returns
    -------
    np.array
    """
    not_missing_idx = np.where( ~np.isnan(sample) )[0]
    to_mask = np.random.choice([False, True], p=[0.5, 0.5])
    if to_mask and len(not_missing_idx) > 2:
        idx_to_mask = np.random.choice( not_missing_idx, size=np.random.choice( np.arange(1, len(not_missing_idx ) -1) ), replace=False)
        sample[ idx_to_mask ] = np.nan
    return sample


if __name__ == "__main__":
    pass
