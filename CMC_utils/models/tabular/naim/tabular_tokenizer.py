import enum
import math
import torch
from torch import Tensor, nn

from typing import List

class _TokenInitialization(enum.Enum):
    UNIFORM = 'uniform'
    NORMAL = 'normal'

    @classmethod
    def from_str(cls, initialization: str) -> '_TokenInitialization':
        try:
            return cls(initialization)
        except ValueError:
            valid_values = [x.value for x in _TokenInitialization]
            raise ValueError(f'initialization must be one of {valid_values}')

    def apply(self, x: Tensor, d: int) -> None:
        d_sqrt_inv = 1 / math.sqrt(d)
        if self == _TokenInitialization.UNIFORM:
            nn.init.uniform_(x, a=-d_sqrt_inv, b=d_sqrt_inv)
        elif self == _TokenInitialization.NORMAL:
            nn.init.normal_(x, std=d_sqrt_inv)


class CategoricalFeatureTokenizer(nn.Module):
    """Transforms categorical features to tokens (embeddings).
    See `FeatureTokenizer` for the illustration.
    The module efficiently implements a collection of `torch.nn.Embedding` (with
    optional biases).
    Examples:
        .. testcode::
            # the input must contain integers. For example, if the first feature can
            # take 3 distinct values, then its cardinality is 3 and the first column
            # must contain values from the range `[0, 1, 2]`.
            cardinalities = [3, 10]
            x = torch.tensor([
                [0, 5],
                [1, 7],
                [0, 2],
                [2, 4]
            ])
            n_objects, n_features = x.shape
            d_token = 3
            tokenizer = CategoricalFeatureTokenizer(cardinalities, d_token, True, 'uniform')
            tokens = tokenizer(x)
            assert tokens.shape == (n_objects, n_features, d_token)
    """

    category_offsets: Tensor

    def __init__(
        self,
        cardinalities: List[int],
        d_token: int,
        bias: bool,
        padding_idx: int = None,
        initialization: str = None,
    ) -> None:
        """
        Args:
            cardinalities: the number of distinct values for each feature. For example,
                :code:`cardinalities=[3, 4]` describes two features: the first one can
                take values in the range :code:`[0, 1, 2]` and the second one can take
                values in the range :code:`[0, 1, 2, 3]`.
            d_token: the size of one token.
            bias: if `True`, for each feature, a trainable vector is added to the
                embedding regardless of feature value. The bias vectors are not shared
                between features.
            initialization: initialization policy for parameters. Must be one of
                :code:`['uniform', 'normal']`. Let :code:`s = d ** -0.5`. Then, the
                corresponding distributions are :code:`Uniform(-s, s)` and :code:`Normal(0, s)`. In
                the paper [gorishniy2021revisiting], the 'uniform' initialization was
                used.
        References:
            * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
        """
        super().__init__()
        assert cardinalities, 'cardinalities must be non-empty'
        assert d_token > 0, 'd_token must be positive'
        if initialization is not None:
            initialization_ = _TokenInitialization.from_str(initialization)
        else:
            initialization_ = None

        category_offsets = torch.tensor([0] + cardinalities[:-1]).cumsum(0)
        self.register_buffer('category_offsets', category_offsets, persistent=False)
        self.padding_idx = padding_idx
        self.embeddings = nn.Embedding(sum(cardinalities), d_token, padding_idx=padding_idx, norm_type=2, max_norm=1)
        self.bias = nn.Parameter(Tensor(len(cardinalities), d_token)) if bias else None

        if initialization is not None:
            for parameter in [self.embeddings.weight, self.bias]:
                if parameter is not None:
                    initialization_.apply(parameter, d_token)
                    self._fill_padding_idx_with_zero()

    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None:
            with torch.no_grad():
                self.embeddings.weight[self.padding_idx].fill_(0)

    @property
    def n_tokens(self) -> int:
        """The number of tokens."""
        return len(self.category_offsets)

    @property
    def d_token(self) -> int:
        """The size of one token."""
        return self.embeddings.embedding_dim

    def forward(self, x: Tensor, value: Tensor = None) -> Tensor:
        x = self.embeddings(x.long() + self.category_offsets[None])

        if value is not None:
            value = value.unsqueeze(0).unsqueeze(2).repeat(1, 1, x.shape[2])
            x = value * x

        if self.bias is not None:
            x = x + self.bias[None]
        return x


if __name__ == "__main__":
    pass
