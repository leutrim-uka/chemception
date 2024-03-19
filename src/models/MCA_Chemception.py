import logging
import sys

import torch
from torch import nn
import torch.nn.functional as F

from layers.ContextualAttentionLayer import ContextualAttentionLayer
from layers.DenseAttentionLayer import DenseAttentionLayer
from layers.EmbeddingLayer import EmbeddingLayer

# setup logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)

class MCA_Chemception(nn.Module):
    def __init__(self, params, *args, **kwargs):
        super(MCA_Chemception, self).__init__(*args, **kwargs)
        


