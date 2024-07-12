import os
import copy
import torch
import logging
import numpy as np
from CMC_utils import save_load, models
from hydra.utils import instantiate
from CMC_utils.miscellaneous import do_nothing, do_really_nothing


log = logging.getLogger(__name__)


__all__ = ["EarlyStopper", "ModelCheckpointer", "CustomScheduler", "TrainManager"]


class EarlyStopper:
    """
    Early stopping to stop the training when the performance or the loss does not improve after certain epochs.
    """
    def __init__(self, patience: int, min_epochs: int, epochs_no_improve: int = 0, early_stop: bool = False):
        self.patience = patience
        self.min_epochs = min_epochs

        self.state = dict( epochs_no_improve = epochs_no_improve, early_stop = early_stop )

    def update_state(self, epoch: int):
        if epoch > self.min_epochs:
            self.state["epochs_no_improve"] += 1
            self.state["early_stop"] = self.state["epochs_no_improve"] >= self.patience


class ModelCheckpointer:
    """
    Model checkpointer to save the best performing model.
    """
    def __init__(self, model, optimizer, best_epoch: int = -1, best_loss: float = None, best_performance: list = None ):
        self.best_model_weights = copy.deepcopy(model.state_dict())
        self.best_optimizer_weights = copy.deepcopy(optimizer.state_dict())
        self.state = dict( best_epoch = best_epoch, best_loss = np.Inf if best_loss is None else best_loss, best_performance = -np.Inf if best_performance is None else best_performance )

    def update_state(self, model, optimizer, epoch: int, epoch_loss: float, epoch_performance: list = None):
        self.best_model_weights = copy.deepcopy(model.state_dict())
        self.best_optimizer_weights = copy.deepcopy(optimizer.state_dict())
        self.state["best_epoch"] = epoch
        self.state["best_loss"] = epoch_loss
        self.state["best_performance"] = epoch_performance


class CustomScheduler:
    """
    Scheduler to reduce the learning rate when the performance or the loss does not improve after certain epochs.
    """
    def __init__(self, learning_rates: list, patience: int, verbose: int, min_epochs: int, epochs_no_improve: int = 0, learning_rate_idx: int = 0):

        self.learning_rates = learning_rates
        self.patience = patience
        self.verbose = verbose
        self.min_epochs = min_epochs

        self.state = dict(epochs_no_improve=epochs_no_improve, learning_rate_idx=learning_rate_idx)

    @property
    def next_step(self):
        return self.state["epochs_no_improve"] >= self.patience and self.state["learning_rate_idx"] < len(self.learning_rates) - 1

    @property
    def lr(self):
        lr_idx = min(self.state["learning_rate_idx"], len(self.learning_rates) - 1)
        return self.learning_rates[lr_idx]

    def update_state(self, epoch: int):
        print_options = {0: do_really_nothing, 1: log.info}

        if epoch > self.min_epochs:
            self.state["epochs_no_improve"] += 1

            if self.next_step:
                self.state["learning_rate_idx"] += 1
                print_options[self.verbose](f"Learning rate reduced to {self.lr}")


class TrainManager:
    """
    Train manager to handle the training and validation process.
    """
    def __init__(self, model, filename: str, path: str, min_epochs: int, max_epochs: int, metrics: dict, early_stop: dict, model_checkpoint: dict, scheduler: dict, optimizer: dict, model_params: dict, performance_tolerance: float = 1e-3, device: str = "cpu", verbose: int = 0, **_ ):
        self.filename = filename
        self.path = path
        self.init_epoch = 0
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.early_stop_params = early_stop
        self.model_checkpoint_params = model_checkpoint
        self.scheduler_params = scheduler
        self.optimizer_params = optimizer
        self.model_params = model_params
        self.tol = performance_tolerance
        self.device = self.set_device(device)
        self.verbose = verbose

        self.history = {phase: {"loss": [], **{metric["name"]: [] for metric in metrics.values()}} for phase in ["train", "val"]}

        self.early_stop = instantiate(self.early_stop_params)
        self.scheduler = instantiate(self.scheduler_params)

        self.optimizer_params = optimizer
        self.optimizer = instantiate(self.optimizer_params["init"], model.parameters(), lr = self.scheduler.lr)

        self.model_checkpoint = instantiate(self.model_checkpoint_params, model=model, optimizer=self.optimizer)

    def load_checkpoint(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Load checkpoint if it exists.
        Parameters
        ----------
        model : torch.nn.Module

        Returns
        -------
        None
        """
        checkpoint_path = os.path.join( self.path, self.filename )
        if os.path.exists(checkpoint_path + "_checkpoint.pkl"):
            checkpoint = save_load.load_model(checkpoint_path + "_checkpoint.pkl")
            optimizer_state_dict = torch.load(checkpoint_path + "_optimizer.pth")
            self.optimizer.load_state_dict(optimizer_state_dict)
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

            self.history = save_load.load_model(checkpoint_path + "_history.pkl")

            self.early_stop = instantiate(self.early_stop_params, **checkpoint.get("early_stop"))
            self.model_checkpoint = instantiate(self.model_checkpoint_params, model=model, optimizer=self.optimizer, **checkpoint.get("model_checkpoint"))
            self.scheduler = instantiate(self.scheduler_params, **checkpoint.get("scheduler"))

            self.init_epoch = checkpoint.get("tr_manager").get("init_epoch", self.init_epoch)

            model_state_dict = torch.load(checkpoint_path + f".{self.model_params['file_extension']}")
            model.load_state_dict(model_state_dict)
        return model

    def epochs(self) -> int:
        """
        Generator to iterate over the epochs.
        Returns
        -------
        int
            Current epoch.
        """
        print_options = {0: do_nothing, 1: log.info}
        for epoch in range(self.init_epoch, self.max_epochs):
            print_options[self.verbose]('-' * 10)
            print_options[self.verbose]('Epoch {}/{}'.format(epoch+1, self.max_epochs))
            yield epoch

    def save_checkpoint(self, epoch) -> None:
        """
        Save checkpoint.
        Parameters
        ----------
        epoch : int

        Returns
        -------
        None
        """
        checkpoint_params = dict(tr_manager=dict(init_epoch=epoch), early_stop=self.early_stop.state, model_checkpoint=self.model_checkpoint.state, scheduler=self.scheduler.state)
        params = dict(self.model_params["init_params"])
        params["best_epoch"] = self.model_checkpoint.state["best_epoch"]
        params["best_loss"] = self.model_checkpoint.state["best_loss"]
        params["best_performance"] = self.model_checkpoint.state["best_performance"]
        params["last_epoch"] = epoch
        params["early_stop"] = self.early_stop.state["early_stop"]

        save_load.save_model(self.model_checkpoint.best_model_weights, self.filename, self.path, extension=self.model_params['file_extension'], model_params=params)
        save_load.save_model(self.model_checkpoint.best_optimizer_weights, self.filename+"_optimizer", self.path, extension="pth")
        save_load.save_model(self.history, self.filename+"_history", self.path, extension="pkl")
        save_load.save_model(checkpoint_params, self.filename+"_checkpoint", self.path, extension="pkl")

    def callbacks(self, model, epoch: int, epoch_loss: float, epoch_performance: float = None) -> None:
        """
        Callbacks to update the state of the early stopper, the model checkpointer and the scheduler.
        Parameters
        ----------
        model : torch.nn.Module
        epoch : int
        epoch_loss : float
        epoch_performance : float

        Returns
        -------
        None
        """
        if epoch <= self.min_epochs:
            performance_improved = False
        elif epoch_performance is not None:
            performance_improved = (epoch_performance - self.model_checkpoint.state["best_performance"]) > self.tol
        else:
            performance_improved = (self.model_checkpoint.state["best_loss"] - epoch_loss) > self.tol

        if performance_improved:
            self.model_checkpoint.update_state(model=model, optimizer=self.optimizer, epoch=epoch, epoch_loss=epoch_loss, epoch_performance=epoch_performance)
            self.early_stop.state["epochs_no_improve"] = 0
            self.scheduler.state["epochs_no_improve"] = 0
            self.save_checkpoint(epoch=epoch)
        else:
            self.early_stop.update_state(epoch=epoch)
            self.scheduler.update_state(epoch=epoch)

            if self.scheduler.next_step:
                self.optimizer = instantiate(self.optimizer_params["init"], model.parameters(), lr=self.scheduler.lr)
                self.scheduler.state["epochs_no_improve"] = 0

    @staticmethod
    def set_device(device: str):
        """
        Set device for the training.
        Parameters
        ----------
        device : str

        Returns
        -------
        torch.device
        """
        return models.set_device(device)

    @staticmethod
    def update_train_params(loss, optimizer):
        """
        Update the parameters of the model.
        Parameters
        ----------
        loss : torch.Tensor
        optimizer : torch.optim.Optimizer

        Returns
        -------
        None
        """
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    pass
