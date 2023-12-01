import torch
import torch.nn as nn


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_log_molar(y, ic50_max=None, ic50_min=None):
    """
    Converts PaccMann predictions from [0,1] to log(micromolar) range.
    """
    return y * (ic50_max - ic50_min) + ic50_min


class Squeeze(nn.Module):
    """Squeeze wrapper for nn.Sequential."""

    def forward(self, data):
        return torch.squeeze(data)


class Unsqueeze(nn.Module):
    """Unsqueeze wrapper for nn.Sequential."""

    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, data):
        return torch.unsqueeze(data, self.dim)


class Temperature(nn.Module):
    """Temperature wrapper for nn.Sequential."""

    def __init__(self, temperature):
        super(Temperature, self).__init__()
        self.temperature = temperature

    def forward(self, data):
        return data / self.temperature
