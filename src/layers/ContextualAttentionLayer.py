import torch
from torch import nn
import torch.nn.functional as F


class ContextualAttentionLayer(nn.Module):
    """
    Inspired by Bahdanau attention. Each token of a SMILES
    targets the genes in a specific way.
    """
    def __init__(self, attention_size):
        super(ContextualAttentionLayer, self).__init__()

        self.attention_size = attention_size

    def forward(self, genes, smiles, reduce_sequence=True, return_alphas=True):
        genes = torch.unsqueeze(genes, 2) if len(genes.shape) == 2 else genes
        hidden_size = smiles.size(2)
        num_genes = genes.size(1)
        num_gene_features = genes.size(2)

        # Trainable parameters
        w_num_gene_features = nn.Parameter(torch.randn(num_gene_features) * 0.1)
        w_genes = nn.Parameter(torch.randn(num_genes, self.attention_size) * 0.1)
        b_genes = nn.Parameter(torch.randn(self.attention_size) * 0.1)
        w_smiles = nn.Parameter(torch.randn(hidden_size, self.attention_size) * 0.1)
        b_smiles = nn.Parameter(torch.randn(self.attention_size) * 0.1)
        v = nn.Parameter(torch.randn(self.attention_size) * 0.1)

        genes_collapsed = torch.tensordot(
            genes, w_num_gene_features, dims=([2], [0])
        )

        x = F.tanh(
            torch.unsqueeze(
                torch.tensordot(genes_collapsed, w_genes, dims=1) + b_genes,
                dim=1
            ) + (torch.tensordot(smiles, w_smiles, dims=1) + b_smiles)
        )

        xv = torch.tensordot(x, v, dims=1)
        alphas = F.softmax(xv, dim=1)

        output = (
            torch.sum(smiles * torch.unsqueeze(alphas, -1), dim=1)
            if reduce_sequence == True
            else smiles * torch.unsqueeze(alphas, -1)
        )

        if return_alphas:
            return output, alphas
        else:
            return output


