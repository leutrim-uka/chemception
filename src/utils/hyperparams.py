import torch.nn as nn
import torch.optim as optim

from loss_functions import (
    mse_cc_loss,
    correlation_coefficient_loss
)

LOSS_FN_FACTORY = {
    'mse': nn.MSELoss(),
    'l1': nn.L1Loss(),
    'mse_and_pearson': mse_cc_loss,
    'pearson': correlation_coefficient_loss,
    'binary_cross_entropy': nn.BCELoss()
}

ACTIVATION_FN_FACTORY = {
    'relu': nn.ReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'tanh': nn.Tanh(),
    'lrelu': nn.LeakyReLU(),
    'elu': nn.ELU()
}

OPTIMIZER_FACTORY = {
    'adam': optim.Adam,
    'adadelta': optim.Adadelta,
    'adagrad': optim.Adagrad,
    'gd': optim.SGD,
    'sparseadam': optim.SparseAdam,
    'adamax': optim.Adamax,
    'asgd': optim.ASGD,
    'lbfgs': optim.LBFGS,
    'rmsprop': optim.RMSprop,
    'rprop': optim.Rprop
}