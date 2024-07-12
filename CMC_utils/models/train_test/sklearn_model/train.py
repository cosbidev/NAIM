import os
import logging
from hydra.utils import call
from CMC_utils.save_load import create_directory
from CMC_utils.miscellaneous import do_nothing
from CMC_utils.datasets import SupervisedTabularDatasetTorch

log = logging.getLogger(__name__)


__all__ = ["train_sklearn_model"]


def train_sklearn_model(model, train_set: SupervisedTabularDatasetTorch, model_params: dict, model_path: str, test_fold: int = 0, val_fold: int = 0, **kwargs) -> None:
    """
    Train a sklearn model
    Parameters
    ----------
    model : sklearn model
    train_set : SupervisedTabularDatasetTorch
    model_params : dict
    model_path : str
    test_fold : int
    val_fold : int
    kwargs : dict

    Returns
    -------
    None
    """
    model_path = os.path.join(model_path, model_params["name"])
    create_directory(model_path)
    filename = f"{test_fold}_{val_fold}"

    if os.path.exists(os.path.join(model_path, filename + f".{model_params['file_extension']}")):
        return

    data, labels, _ = train_set.get_data()

    labels_options = dict(binary=do_nothing, discrete=do_nothing)
    labels = labels_options[train_set.label_type](labels)

    model.fit(data, labels, **model_params["fit_params"])

    call(model_params["save_function"], model, filename, model_path, model_params=model_params["init_params"], extension=model_params["file_extension"], _recursive_=False)

    log.info("Model trained")


if __name__ == "__main__":
    pass
