import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LazyLinear


class DenseAttentionLayer(nn.Module):
    def __init__(self, return_alphas=False):
        super(DenseAttentionLayer, self).__init__()
        self.return_alphas = return_alphas

        self.feature_collapse = LazyLinear(1)
        self.attention = LazyLinear(None)

    def forward(self, inputs):
        if inputs.dim() == 3:
            inputs = torch.squeeze(
                F.relu(self.feature_collapse(inputs)),
                dim=2
            )

        assert inputs.dim() == 2

        alphas = F.softmax(
            self.attention(inputs),
            dim=1

        )
        output = inputs * alphas

        return (output, alphas) if self.return_alphas else output
