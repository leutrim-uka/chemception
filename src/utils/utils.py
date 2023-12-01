import torch

LOSS_FN_FACTORY = {
    'mse': nn.MSELoss(),
    'l1': nn.L1Loss(),
    'mse_and_pearson': mse_cc_loss,
    'pearson': correlation_coefficient_loss,
    'binary_cross_entropy': nn.BCELoss()
}

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')