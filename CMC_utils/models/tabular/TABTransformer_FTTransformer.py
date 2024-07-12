import torch
import numpy as np
from hydra.utils import instantiate
from tab_transformer_pytorch import TabTransformer as TABT, FTTransformer as FTT

__all__ = ["TABTransformer", "FTTransformer"]

class TABTransformer(torch.nn.Module):
    def __init__(self, cat_idxs: list = None, **kwargs):
        super().__init__()
        self.cat_idxs = cat_idxs
        kwargs["mlp_act"] = instantiate(kwargs["mlp_act"])
        self.model = TABT(**kwargs)

    def forward(self, inputs):
        n_features = inputs.shape[-1]
        cat_features_map = torch.BoolTensor(np.array(list(map(lambda x: x in self.cat_idxs, range(n_features))), dtype=bool) ).to(inputs.device)
        x_cat = inputs[:, cat_features_map].type(torch.LongTensor).to(inputs.device)
        x_cont = inputs[:, ~cat_features_map].to(inputs.device)
        return self.model(x_cat, x_cont)


class FTTransformer(torch.nn.Module):
    def __init__(self, cat_idxs: list = None, **kwargs):
        super().__init__()
        self.cat_idxs = cat_idxs
        self.model = FTT(**kwargs)

    def forward(self, inputs):
        n_features = inputs.shape[-1]
        cat_features_map = torch.BoolTensor(np.array(list(map(lambda x: x in self.cat_idxs, range(n_features))), dtype=bool)).to(inputs.device)
        x_cat = inputs[:, cat_features_map].type(torch.LongTensor).to(inputs.device)
        x_cont = inputs[:, ~cat_features_map].to(inputs.device)
        return self.model(x_cat, x_cont)


if __name__ == "__main__":
    pass
