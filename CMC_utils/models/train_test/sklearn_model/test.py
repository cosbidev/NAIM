import os
import logging
import pandas as pd
from hydra.utils import call
from CMC_utils.save_load import save_table
from CMC_utils.preprocessing import discrete_to_label
from CMC_utils.datasets import SupervisedTabularDatasetTorch

log = logging.getLogger(__name__)

__all__ = ["test_sklearn_model"]


def test_sklearn_model(*sets: SupervisedTabularDatasetTorch, model_params: dict, model_path: str, prediction_path: str, test_fold: int = 0, val_fold: int = 0, **kwargs) -> None:
    """
    Test a sklearn model
    Parameters
    ----------
    sets : SupervisedTabularDatasetTorch
    model_params : dict
    model_path : str
    prediction_path : str
    test_fold : int
    val_fold : int
    kwargs : dict

    Returns
    -------
    None
    """
    model_path = os.path.join( model_path, model_params["name"], f"{test_fold}_{val_fold}.{model_params['file_extension']}")
    model = call(model_params["load_function"], model_path)

    for fset in sets:

        filename = f"{test_fold}_{val_fold}_{fset.set_name}.csv"
        if os.path.exists(os.path.join(prediction_path, filename)):
            continue

        data, labels, ID = fset.get_data()

        probs = model.predict_proba(data.astype(float))

        labels = list(map( lambda label: discrete_to_label(label, classes=tuple(sets[0].classes)), labels))

        preds = list(map( lambda label: discrete_to_label(label, classes=tuple(sets[0].classes)), probs))

        if fset.label_type == "binary":
            probs = probs[:, 1]

        results = pd.DataFrame( dict( ID=ID, label=labels, prediction=preds, probability=probs.tolist() ))
        save_table( results, filename, prediction_path )
        log.info("Inference done")


if __name__ == "__main__":
    pass
