import torch
from torch.nn import Module

__all__ = ["L1RegularizationLoss", "L2RegularizationLoss"]


class L1RegularizationLoss(Module):
    """
    L1 Regularization Loss
    """
    def __init__(self):
        super(L1RegularizationLoss, self).__init__()

    @staticmethod
    def forward(model):
        # Compute L1 loss component

        l1_parameters = []
        for parameter in model.parameters():
            l1_parameters.append(parameter.view(-1))
        l1 = torch.abs( torch.cat(l1_parameters) ).sum()  # self.alpha *

        return l1


class L2RegularizationLoss(Module):
    """
    L2 Regularization Loss
    """
    def __init__(self):
        super(L2RegularizationLoss, self).__init__()

    @staticmethod
    def forward(model):
        # Compute L1 loss component

        l2_parameters = []
        for parameter in model.parameters():
            l2_parameters.append(parameter.view(-1))
        l2 = torch.abs( torch.cat(l2_parameters) ).sum()  # self.alpha *

        return l2


if __name__ == "__main__":
    pass
