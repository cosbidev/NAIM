import logging
import numpy as np
import pandas as pd
from abc import abstractmethod
from typing import List, Tuple

from CMC_utils.preprocessing.labels import *
from .generic import TabularDatasetSupervised

__all__ = ["SupervisedTaskDataset", "ClassificationDataset"]

log = logging.getLogger(__name__)


class SupervisedTaskDataset(TabularDatasetSupervised):
    """
    Class for supervised datasets.
    """
    __label_type: str = None
    __model_label_types: List[str] = None

    def __init__(self, **kwargs):
        TabularDatasetSupervised.__init__(self, **kwargs)

    @abstractmethod
    def __set_label_type(self, label_type, model_label_types, model_framework):
        self.__label_type = label_type

    @abstractmethod
    def __set_label_encoding_functions(self):
        self.__label_encode = lambda label: label
        self.__label_decode = lambda label: label

        self.__set_label_encode_signature()

    @abstractmethod
    def __set_label_encode_signature(self):
        input_signature = ''  # + ('n' * (self.__label_type in ("single_risk_survival", "competing_risks_survival")))
        output_signature = ''  # + ('m' * (self.__label_type not in ('binary', 'discrete')))

        self.__label_encode_signature = f"({input_signature})->({output_signature})"

    @abstractmethod
    def __compute_info_for_cv(self):
        self.__info_for_cv = self.__labels

    @abstractmethod
    def __compute_labels_for_model(self):
        self.__labels_for_model = self.batch_encode(self.labels)

    @abstractmethod
    def label_encode( self, label ):
        encoded_label = self.__label_encode( label )
        return encoded_label

    @abstractmethod
    def label_decode( self, label ):
        decoded_label = self.__label_decode( label )
        return decoded_label

    @abstractmethod
    def batch_encode(self, labels):
        label_function = np.vectorize(self.__label_encode, signature=self.__label_encode_signature)
        return label_function( labels )


class ClassificationDataset(SupervisedTaskDataset):
    """
    Class for classification datasets.
    """
    def __init__(self, label_type: str, model_label_types: List[str], model_framework: str, classes: Tuple[str], **kwargs):
        SupervisedTaskDataset.__init__(self, **kwargs)

        self.__classes = classes
        self.__set_label_type(label_type, model_label_types, model_framework)
        self.__set_label_encoding_functions()
        self.__compute_info_for_cv()
        self.__compute_labels_for_model()

        self.preprocessing_params["label_type"] = self.label_type
        self.preprocessing_params["classes"] = self.classes
        log.info("Dataset ready")

    @property
    def classes(self):
        return self.__classes

    @property
    def label_type(self):
        return self.__label_type

    @property
    def info_for_cv(self):
        return self.__info_for_cv

    @property
    def labels_for_model(self):
        return self.__labels_for_model

    def __set_label_type(self, label_type, model_label_types, model_framework):
        if label_type == "multiclass":
            label_type = "categorical"
            label_type = label_type if label_type in model_label_types else "discrete"
        self.__label_type = label_type

    def __set_label_encoding_functions(self):
        if self.__label_type in ("binary", "discrete"):
            self.__label_encode = lambda label: label_to_discrete(label, self.__classes)
            self.__label_decode = lambda label: discrete_to_label(label, self.__classes)
        elif self.__label_type == "categorical":
            self.__label_encode = lambda label: label_to_categorical(label, self.__classes)
            self.__label_decode = lambda label: categorical_to_label(label, self.__classes)
        else:
            raise Exception("Sorry, desired label encoding does not exist.")
        self.__set_label_encode_signature()

    def __set_label_encode_signature(self):
        input_signature = ''
        output_signature = '' + ('m' * (self.__label_type == "categorical"))

        self.__label_encode_signature = f"({input_signature})->({output_signature})"

    def __compute_info_for_cv(self):
        self.__info_for_cv = self.labels.rename({"label": "target"}, axis=1)

    def __compute_labels_for_model(self):

        labels_for_model = self.batch_encode(self.labels)  # .drop("ID", axis=1))

        if self.__label_type == "categorical":
            labels_for_model = np.squeeze(labels_for_model)

        self.__labels_for_model = pd.DataFrame(labels_for_model, index=self.labels.index)

    def label_encode( self, label ):
        encoded_label = self.__label_encode( label )
        return encoded_label

    def label_decode( self, label ):
        decoded_label = self.__label_decode( label )
        return decoded_label

    def batch_encode(self, labels):
        label_function = np.vectorize(self.__label_encode, signature=self.__label_encode_signature)
        return label_function( labels )


if __name__ == "__main__":
    pass
