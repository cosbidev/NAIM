import os
import torch
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Union
import torch.nn.functional as F
from omegaconf import DictConfig
from hydra.utils import instantiate, call
from torch.utils.data import DataLoader
from CMC_utils.save_load import save_table
from CMC_utils.models import set_device
from CMC_utils.datasets import SupervisedTabularDatasetTorch

from .outputs_functions import *

log = logging.getLogger(__name__)

__all__ = ["test_torch_model"]


def test_torch_model(*sets: SupervisedTabularDatasetTorch, model_params: dict, model_path: str, prediction_path: str, train_params: Union[dict, DictConfig], test_fold: int = 0, val_fold: int = 0, **kwargs) -> None:
    """
    Test a torch model
    Parameters
    ----------
    sets : SupervisedTabularDatasetTorch
    model_params : dict
    model_path : str
    prediction_path : str
    train_params : Union[dict, DictConfig]
    test_fold : int
    val_fold : int
    kwargs : dict

    Returns
    -------
    None
    """
    sets = [ DataLoader(fset, batch_size=train_params.dl_params.batch_size, shuffle=False, drop_last=False) for fset in sets ]

    model_path = os.path.join(model_path, model_params["name"], f"{test_fold}_{val_fold}.{model_params['file_extension']}")

    device = set_device(train_params.dl_params.device)

    model = instantiate(model_params["init_params"], _recursive_=False)

    model.load_state_dict(call(model_params["load_function"], model_path))
    model = model.to(device)

    output_to_pred_options = {"binary": surpass_threshold, "categorical": max_index}

    model.eval()

    for fset in sets:

        filename = f"{test_fold}_{val_fold}_{fset.dataset.set_name}.csv"
        if os.path.exists(os.path.join(prediction_path, filename)):
            continue

        pbar = tqdm(fset, leave=False, disable=(not train_params.dl_params.verbose_batch))

        total_results = pd.DataFrame()

        with torch.set_grad_enabled(False):
            for *input_list, labels, idxs in pbar:
                input_list = [inputs.float().to(device) for inputs in input_list]
                labels = labels.to(device)

                outputs = model(*[inputs.float() for inputs in input_list])

                if (fset.dataset.label_type == "binary" and outputs.shape[-1] == 2) or fset.dataset.label_type == "categorical":
                    outputs = F.softmax(outputs, dim=-1)

                preds = output_to_pred_options[fset.dataset.label_type](outputs, return_first=True, **kwargs)

                if labels.shape != preds.shape:
                    labels = output_to_pred_options[fset.dataset.label_type](labels, return_first=True, **kwargs)

                idxs = np.array(idxs)
                labels = np.squeeze(labels.cpu().detach().numpy()).astype(int)
                preds = np.squeeze(preds.cpu().detach().numpy()).astype(int)
                outputs = outputs.cpu().detach().numpy().astype(float)

                if fset.dataset.label_type == "binary":
                    if outputs.shape[1] == 2:
                        outputs = outputs[:, 1]
                    outputs = np.squeeze(outputs)

                running_results = pd.DataFrame(dict( ID=idxs, label=labels.tolist(), prediction=preds.tolist(), probability=outputs.tolist()))
                total_results = pd.concat( [total_results, running_results], axis=0, ignore_index=True)

        save_table(total_results, filename, prediction_path)
        log.info("Inference done")


if __name__ == "__main__":
    pass
