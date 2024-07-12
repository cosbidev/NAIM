import os
import torch
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Union, Tuple
from omegaconf import DictConfig
from hydra.utils import call, instantiate
from sklearn.utils.class_weight import compute_sample_weight
from torch.utils.data import DataLoader, WeightedRandomSampler

from .outputs_functions import *

import torch.nn.functional as F
from CMC_utils.save_load import create_directory
from CMC_utils.models import initialize_weights
from CMC_utils.metrics import metrics_computation_df
from CMC_utils.datasets import SupervisedTabularDatasetTorch
from CMC_utils.miscellaneous import do_really_nothing, seed_worker

log = logging.getLogger(__name__)

__all__ = ["train_torch_model"]


def train_torch_model(model: torch.nn.Module, train_set: SupervisedTabularDatasetTorch, model_params: dict, model_path: str, val_set: SupervisedTabularDatasetTorch, train_params: Union[dict, DictConfig], test_fold: int = 0, val_fold: int = 0, use_weights: bool = True, **kwargs) -> None:
    """
    Train a torch model
    Parameters
    ----------
    model : torch.nn.Module
    train_set : SupervisedTabularDatasetTorch
    model_params : dict
    model_path : str
    val_set : SupervisedTabularDatasetTorch
    train_params : Union[dict, DictConfig]
    test_fold : int
    val_fold : int
    use_weights : bool
    kwargs : dict

    Returns
    -------
    None
    """
    if use_weights:
        labels = np.argmax(train_set.labels, axis=1)

        weights = compute_sample_weight(class_weight = "balanced", y = labels)
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(train_set), replacement=False)
        train_dataloader = DataLoader(train_set, batch_size=train_params.dl_params.batch_size, sampler=sampler, drop_last=True, worker_init_fn=seed_worker)
    else:
        train_dataloader = DataLoader(train_set, batch_size=train_params.dl_params.batch_size, shuffle=True, drop_last=True, worker_init_fn=seed_worker)

    val_dataloader = DataLoader(val_set, batch_size=train_params.dl_params.batch_size, shuffle=False, drop_last=False)
    dataloaders = dict(train=train_dataloader, val=val_dataloader)

    initialize_weights(model, train_params.initializer)

    metrics = train_params.get("set_metrics", {})
    model_path = os.path.join(model_path, model_params["name"])
    create_directory(model_path)
    filename = f"{test_fold}_{val_fold}"
    tr_manager = instantiate(train_params["manager"], model, filename=filename, path=model_path, metrics=metrics, optimizer=train_params["optimizer"], model_params=model_params, _recursive_=False)

    model = tr_manager.load_checkpoint(model)
    model = model.to(tr_manager.device)

    callback_options = {True: tr_manager.callbacks, False: do_really_nothing}
    print_options = {0: do_really_nothing, 1: log.info}

    epoch, phase, performance = -1, None, None

    if tr_manager.early_stop.state["early_stop"]:
        return

    losses_params = {loss_name: call(loss_params.set_params_function, loss_params, model_params=model_params, _recursive_=False) for loss_name, loss_params in train_params.loss.items()}
    criterions = [instantiate(loss_params["init_params"]).to(tr_manager.device) for loss_params in losses_params.values()]
    regularizers = [instantiate(regularizer_params["init_params"]).to(tr_manager.device) for regularizer_params in train_params.regularizer.values()]

    for epoch in tr_manager.epochs():

        if tr_manager.early_stop.state["early_stop"]:
            break

        optimizer = tr_manager.optimizer
        for phase, dataloader in dataloaders.items():
            model, epoch_loss, epoch_results = model_predict(model, dataloader, criterions, regularizers, optimizer, tr_manager, train_params=train_params, phase=phase, **kwargs )

            epoch_performance = metrics_computation_df(epoch_results, metrics=metrics, classes=dataloader.dataset.classes, use_weights=True, verbose=0)

            print_options[tr_manager.verbose]('{} loss: {:.4f} '.format(phase, epoch_loss) + " ".join([f"{metric}: {epoch_performance[metric].values.tolist()}" for metric in epoch_performance.columns]))

            tr_manager.history[phase]['loss'].append(epoch_loss)
            for metric in epoch_performance.columns:
                performance = epoch_performance[metric].values.tolist()
                tr_manager.history[phase][metric].append(performance)
                performance = np.mean(performance)

            callback_options[phase == "val"](model, epoch, epoch_loss, performance)

    if phase is not None:
        tr_manager.save_checkpoint(epoch + (not tr_manager.early_stop.state["early_stop"]))

        log.info("Model trained")
        print_options[tr_manager.verbose and phase == "val"](tr_manager.model_checkpoint.state)


def model_predict(model, dataloader, criterions, regularizers, optimizer, tr_manager, train_params, phase: str = "val", **kwargs) -> Tuple[torch.nn.Module, float, pd.DataFrame]:
    """
    Predict with a torch model during training
    Parameters
    ----------
    model : torch.nn.Module
    dataloader : DataLoader
    criterions : List[torch.nn.Module]
    regularizers : List[torch.nn.Module]
    optimizer : torch.optim.Optimizer
    tr_manager : TrainManager
    train_params : Union[dict, DictConfig]
    phase : str
    kwargs : dict

    Returns
    -------
    Tuple[torch.nn.Module, float, pd.DataFrame]
    """
    output_to_pred_options = {"binary": surpass_threshold, "categorical": max_index}
    model_options = {True: model.train, False: model.eval}
    train_params_options = {True: tr_manager.update_train_params, False: do_really_nothing}

    running_loss = 0.0
    total_results = pd.DataFrame()

    model_options[phase == "train"]()

    pbar = tqdm(dataloader, leave=False, disable=(not train_params.dl_params.verbose_batch))
    with torch.set_grad_enabled(phase == 'train'):
        for *input_list, labels, idxs in pbar:
            input_list = [inputs.float().to(tr_manager.device) for inputs in input_list]
            labels = labels.to(tr_manager.device)

            optimizer.zero_grad()

            outputs = model(*[inputs.float() for inputs in input_list])

            loss = 0
            for loss_params, criterion in zip( list(train_params.loss.values()), criterions):
                loss += loss_params.alpha * criterion(outputs, labels.float())

            if regularizers:
                for regularizer_params, regularizer in zip(list(train_params.regularizer.values()), regularizers):
                    loss += regularizer_params.alpha * regularizer(model)

            train_params_options[phase == "train"](loss, optimizer)

            if (dataloader.dataset.label_type == "binary" and outputs.shape[-1] == 2) or dataloader.dataset.label_type == "categorical":
                outputs = F.softmax(outputs, dim=-1)

            preds = output_to_pred_options[dataloader.dataset.label_type](outputs, return_first=True, **kwargs)

            if labels.shape != preds.shape:
                labels = output_to_pred_options[dataloader.dataset.label_type](labels, return_first=True, **kwargs)

            idxs = np.array(idxs)
            labels = np.squeeze(labels.cpu().detach().numpy()).astype(int)
            preds = np.squeeze(preds.cpu().detach().numpy()).astype(int)
            outputs = outputs.cpu().detach().numpy().astype(float)

            if dataloader.dataset.label_type == "binary":
                if outputs.shape[1] == 2:
                    outputs = outputs[:, 1]
                outputs = np.squeeze(outputs)

            batch_results = pd.DataFrame(dict( ID=idxs, label=labels.tolist(), prediction=preds.tolist(), probability=outputs.tolist()))
            total_results = pd.concat( [total_results, batch_results], axis=0, ignore_index=True)

            running_loss += loss.item() * input_list[0].size(0)

            pbar.set_postfix_str( "loss {:.4f}, ".format(running_loss / total_results.shape[0]) )

    running_loss = running_loss / total_results.shape[0]

    return model, running_loss, total_results


if __name__ == "__main__":
    pass
