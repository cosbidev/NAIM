from omegaconf import DictConfig
from CMC_utils.save_load import load_tabular_dataset
from CMC_utils.preprocessing import get_preprocessing_params


__all__ = ["Dataset", "TabularDatasetSupervised"]


class Dataset:
    """
    Generic class for datasets
    """
    def __init__(self, name: str, db_type: str, task: str, **_):
        self.__name = name
        self.__db_type = db_type
        self.__task = task

    @property
    def name(self):
        return self.__name

    @property
    def db_type(self):
        return self.__db_type

    @property
    def task(self):
        return self.__task


class TabularDatasetSupervised(Dataset):
    """
    Generic class for tabular datasets
    """
    def __init__(self, name: str, db_type: str, task: str, preprocessing_params: DictConfig, **kwargs):
        """
        This class is used to load tabular datasets
        Parameters
        ----------
        name : str
        db_type : str
        task : str
        path : str
        preprocessing_params : DictConfig
        id_column : str
        data_filename : str
        labels_filename : str
        types_filename : str
        kwargs : dict
        """
        Dataset.__init__(self, name=name, db_type=db_type, task=task)

        self.__load_dataset(**kwargs)

        self.__compute_parameters(preprocessing_params=preprocessing_params)

    @property
    def data(self):
        return self.__data.loc[:, self.__features_types.index.to_list()].copy()

    @property
    def complete_idxs_data(self):
        return self.__data.copy()

    @property
    def labels(self):
        other_idxs_cols = [col for col in self.__data.columns if col not in self.__features_types.index.to_list()]
        if other_idxs_cols:
            return self.__labels.drop(other_idxs_cols, axis=1).copy()
        else:
            return self.__labels.copy()

    @property
    def complete_idxs_labels(self):
        return self.__labels.copy()

    @property
    def feature_types(self):
        return self.__features_types.copy()

    def __load_dataset(self, **kwargs):
        data, labels, features_types = load_tabular_dataset(**kwargs)

        self.__data = data
        self.__labels = labels
        self.__features_types = features_types

    def __compute_parameters(self, preprocessing_params: DictConfig):
        self.preprocessing_params = get_preprocessing_params(self.data, self.feature_types, preprocessing_params)


if __name__ == "__main__":
    pass
