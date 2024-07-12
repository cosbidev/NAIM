import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from CMC_utils.data_augmentation import missing_augmentation
from CMC_utils.preprocessing import set_fold_preprocessing, apply_preprocessing

__all__ = ["SupervisedTabularDatasetTorch"]


class SupervisedTabularDatasetTorch(Dataset):
    """
    Supervised tabular dataset for PyTorch
    """
    def __init__(self, data: pd.DataFrame, labels: pd.DataFrame, set_name: str, preprocessing_params: dict, preprocessing_paths: dict, test_fold: int = 0, val_fold: int = 0, augmentation: bool = False, **kwargs):
        if set_name == "train":
            set_fold_preprocessing(data, test_fold, val_fold, preprocessing_paths, preprocessing_params, **kwargs)

        [data] = apply_preprocessing(data, preprocessing_paths=preprocessing_paths, preprocessing_params=preprocessing_params, test_fold=test_fold, val_fold=val_fold)

        self.__ID = data.index.to_list()
        self.__data = data.values
        self.__labels = labels.values
        self.__columns = data.columns

        self.set_name = set_name
        self.classes = preprocessing_params["classes"]
        self.label_type = preprocessing_params["label_type"]

        self.input_size = self.__data.shape[1]
        self.output_size = self.__labels.shape[1]

        self.augmentation = augmentation

    @property
    def ID(self):
        return self.__ID

    @property
    def data(self):
        return self.__data

    @property
    def labels(self):
        return self.__labels

    @property
    def columns(self):
        return self.__columns

    def __len__(self):
        return self.__data.shape[0]

    def __getitem__(self, index):
        ID = self.__ID[index]
        sample = self.__data[index]
        label = self.__labels[index]

        sample = np.array( sample, dtype=float )

        if self.augmentation:
            sample = missing_augmentation(sample.copy())

        return sample, label, ID

    def get_data(self):
        IDs = self.__ID
        data = self.__data
        labels = np.squeeze(self.__labels)

        return data, labels, IDs


if __name__ == "__main__":
    pass
