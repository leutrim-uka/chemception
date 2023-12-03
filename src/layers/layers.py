import torch
import torch.nn as nn
from collections import OrderedDict
from utils.utils import Unsqueeze, Squeeze


def dense_layer(
    input_size, hidden_size, act_fn=nn.ReLU(), batch_norm=False, dropout=0.0
):
    return nn.Sequential(
        OrderedDict(
            [
                ('projection', nn.Linear(input_size, hidden_size)),
                (
                    'batch_norm',
                    nn.BatchNorm1d(hidden_size)
                    if batch_norm else nn.Identity(),
                ),
                ('act_fn', act_fn),
                ('dropout', nn.Dropout(p=dropout)),
            ]
        )
    )


def convolutional_layer(
    num_kernel,
    kernel_size,
    act_fn=nn.ReLU(),
    batch_norm=False,
    dropout=0.0,
    input_channels=1,
):
    """Convolutional layer.

    Args:
        num_kernel (int): Number of convolution kernels.
        kernel_size (tuple[int, int]): Size of the convolution kernels.
        act_fn (callable): Functional of the nonlinear activation.
        batch_norm (bool): whether batch normalization is applied.
        dropout (float): Probability for each input value to be 0.
        input_channels (int): Number of input channels (defaults to 1).

    Returns:
        callable: a function that can be called with inputs.
    """
    return nn.Sequential(
        OrderedDict(
            [
                (
                    'convolve',
                    torch.nn.Conv2d(
                        input_channels,  # channel_in
                        num_kernel,  # channel_out
                        kernel_size,  # kernel_size
                        # TODO: Potential error. Check utils/layers.py in repo
                        padding=(kernel_size[0] // 2,
                                 0),  # pad for valid conv.
                    ),
                ),

                # Squeeze is a custom class defined in utils.py. It is a wrapper instead of a function.
                ('squeeze', Squeeze()),
                ('act_fn', act_fn),
                ('dropout', nn.Dropout(p=dropout)),
                (
                    'batch_norm',
                    nn.BatchNorm1d(num_kernel)
                    if batch_norm else nn.Identity(),
                ),
            ]
        )
    )