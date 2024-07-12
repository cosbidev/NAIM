import torch.nn
from torch.nn.init import *
from typing import Union, List
from torch.nn import Sequential, ModuleList, Conv2d, BatchNorm2d, Linear, Module

__all__ = ["list_model_modules", "initialize_weights", "set_device"]


def list_model_modules(model: Union[Module, ModuleList, Sequential]) -> List[Module]:
    """
    List all modules in a model
    Parameters
    ----------
    model : Union[Module, ModuleList, Sequential]    model to list modules

    Returns
    -------
    List[Module]   list of modules
    """
    modules = []
    for module in model.children():
        if isinstance(module, (Sequential, ModuleList)):
            modules.extend(list_model_modules(module))
        else:
            modules.append(module)
    return modules


def initialize_weights(model: Union[Module, ModuleList, Sequential], initializer) -> None:
    """
    Initialize weights of a model
    Parameters
    ----------
    model : Union[Module, ModuleList, Sequential]     model to initialize
    initializer : Union[Dict, str]      dictionary with name and params of the initializer or string with the name of the initializer

    Returns
    -------
    None
    """
    initializer_options = dict(uniform=uniform_,
                               normal=normal_,
                               constant=constant_,
                               ones=ones_,
                               zeros=zeros_,
                               eye=eye_,
                               dirac=dirac_,
                               xavier_normal=xavier_normal_,
                               xavier_uniform=xavier_uniform_,
                               kaiming_uniform=kaiming_uniform_,
                               kaiming_normal=kaiming_normal_,
                               trunc_normal=trunc_normal_,
                               orthogonal=orthogonal_,
                               sparse=sparse_)
    for m in model.modules():
        if isinstance(m, Conv2d):
            initializer_options[initializer.name](m.weight, **initializer.params)
            if m.bias is not None:
                constant_(m.bias, 0)
        elif isinstance(m, BatchNorm2d):
            constant_(m.weight, 1)
            constant_(m.bias, 0)
        elif isinstance(m, Linear):
            initializer_options[initializer.name](m.weight, **initializer.params)

            if m.bias is not None:
                constant_(m.bias, 0)


def set_device(device: str) -> torch.device:
    """
    Set device for training
    Parameters
    ----------
    device : str   device to use

    Returns
    -------
    torch.device  device to use
    """
    device_options = {"cpu": True, "cuda": torch.cuda.is_available()}  # "mps": torch.has_mps,
    device = device_options[device] * device + (not device_options[device]) * "cpu"
    device = torch.device(device)
    return device


if __name__ == "__main__":
    pass
