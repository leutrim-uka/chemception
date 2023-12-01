import torch
import torch.nn as nn

from utils.utils import get_device, get_log_molar
from utils.hyperparams import LOSS_FN_FACTORY, ACTIVATION_FN_FACTORY

from collections import OrderedDict
from layers.layers import convolutional_layer, dense_layer

from layers.ContextualAttentionLayer import ContextualAttentionLayer


class MCAv2(nn.Module):
    def __init__(self, params, *args, **kwargs):
        super(MCAv2, self).__init__()

        self.device = get_device()
        self.params = params

        # Set default loss function to MSE
        self.loss_fn = LOSS_FN_FACTORY[params.get('loss_fn', 'mse')]

        self.min_max_scaling = True if params.get(
            'drug_sensitivity_processing_parameters', {}
        ) != {} else False

        # 'min' and 'max' are provided in the params dictionary
        if self.min_max_scaling:
            self.IC50_max = params[
                'drug_sensitivity_processing_parameters'
            ]['parameters']['max']  # yapf: disable
            self.IC50_min = params[
                'drug_sensitivity_processing_parameters'
            ]['parameters']['min']  # yapf: disable

        self.smiles_padding_length = params['smiles_padding_length']
        self.number_of_genes = params.get('number_of_genes', 2128)
        self.smiles_attention_size = params.get('smiles_attention_size', 64)
        self.gene_attention_size = params.get('gene_attention_size', 1)
        self.molecule_temperature = params.get('molecule_temperature', 1.)
        self.gene_temperature = params.get('gene_temperature', 1.)

        # Model architecture (hyperparameter)
        self.molecule_heads = params.get('molecule_heads', [4, 4, 4, 4])
        self.gene_heads = params.get('gene_heads', [2, 2, 2, 2])

        if len(self.gene_heads) != len(self.molecule_heads):
            raise ValueError('Length of gene and molecule_heads do not match.')

        self.filters = params.get('filters', [64, 64, 64])

        self.hidden_sizes = (
                [
                    self.molecule_heads[0] * params['smiles_embedding_size'] + sum(
                        [
                            h * f
                            for h, f in zip(self.molecule_heads[1:], self.filters)
                        ]
                    ) + sum(self.gene_heads) * self.number_of_genes
                ] + params.get('stacked_dense_hidden_sizes', [1024, 512])
        )

        self.dropout = params.get('dropout', 0.5)
        self.temperature = params.get('temperature', 1.)
        self.act_fn = ACTIVATION_FN_FACTORY[
            params.get('activation_fn', 'relu')
        ]
        self.kernel_sizes = params.get(
            'kernel_sizes', [
                [3, params['smiles_embedding_size']],
                [5, params['smiles_embedding_size']],
                [11, params['smiles_embedding_size']],
            ]
        )

        if len(self.filters) != len(self.kernel_sizes):
            raise ValueError(
                'Length of filter and kernel size lists do not match.'
            )

        if len(self.filters) + 1 != len(self.molecule_heads):
            raise ValueError(
                'Length of filter and multihead lists do not match'
            )

        self.smiles_embedding = nn.Embedding(
            self.params['smiles_vocabulary_size'],
            self.params['smiles_embedding_size'],
            scale_grad_by_freq=params.get('embed_scale_grad', False)
        )

        # OrderedDict remembers the order in which the contents are added. it is used to keep layers in the order they
        # were defined
        self.convolutional_layer = nn.Sequential(
            OrderedDict(
                # This is a list of tuples created with list comprehension. Each tuple contains a string (name of layer)
                # and a convolutional layer
                [
                    (
                        # This is a unique name for each convolutional layer
                        f'convolutional_{index}',
                        convolutional_layer(
                            num_kernel,
                            kernel_size,
                            act_fn=self.act_fn,
                            batch_norm=params.get('batch_norm', False),
                            dropout=self.dropout
                        ).to(self.device)
                    ) for index, (num_kernel, kernel_size) in
                    enumerate(zip(self.filters, self.kernel_sizes))
                ]
            )
        )

        smiles_hidden_sizes = [params['smiles_embedding_size']] + self.filters

        self.molecule_attention_layers = nn.Sequential(OrderedDict([
            (
                f'molecule_attention_{layer}_head_{head}',
                ContextualAttentionLayer(
                    reference_hidden_size=smiles_hidden_sizes[layer],
                    reference_sequence_length=self.smiles_padding_length,
                    context_hidden_size=1,
                    context_sequence_length=self.number_of_genes,
                    attention_size=self.smiles_attention_size,
                    individual_nonlinearity=params.get(
                        'context_nonlinearity', nn.Sequential()
                    ),
                    temperature=self.molecule_temperature
                )
            ) for layer in range(len(self.molecule_heads))
            for head in range(self.molecule_heads[layer])
        ]))  # yapf: disable

        # Only applied if params['batch_norm'] = True
        self.batch_norm = nn.BatchNorm1d(self.hidden_sizes[0])
        self.dense_layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        'dense_{}'.format(ind),
                        dense_layer(
                            self.hidden_sizes[ind],
                            self.hidden_sizes[ind + 1],
                            act_fn=self.act_fn,
                            dropout=self.dropout,
                            batch_norm=params.get('batch_norm', True)
                        ).to(self.device)
                    ) for ind in range(len(self.hidden_sizes) - 1)
                ]
            )
        )

        # Gene attention stream
        self.gene_attention_layers = nn.Sequential(OrderedDict([
            (
                f'gene_attention_{layer}_head_{head}',
                ContextualAttentionLayer(
                    reference_hidden_size=1,
                    reference_sequence_length=self.number_of_genes,
                    context_hidden_size=smiles_hidden_sizes[layer],
                    context_sequence_length=self.smiles_padding_length,
                    attention_size=self.gene_attention_size,
                    individual_nonlinearity=params.get(
                        'context_nonlinearity', nn.Sequential()
                    ),
                    temperature=self.gene_temperature
                )
            ) for layer in range(len(self.molecule_heads))
            for head in range(self.gene_heads[layer])
        ]))  # yapf: disable

        # Only applied if params['batch_norm'] = True
        self.batch_norm = nn.BatchNorm1d(self.hidden_sizes[0])
        self.dense_layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        'dense_{}'.format(ind),
                        dense_layer(
                            self.hidden_sizes[ind],
                            self.hidden_sizes[ind + 1],
                            act_fn=self.act_fn,
                            dropout=self.dropout,
                            batch_norm=params.get('batch_norm', True)
                        ).to(self.device)
                    ) for ind in range(len(self.hidden_sizes) - 1)
                ]
            )
        )

        # These are the hidden layers of the dense NN at the end
        self.final_dense = (
            nn.Linear(self.hidden_sizes[-1], 1)
            if not params.get('final_activation', False) else nn.Sequential(
                OrderedDict(
                    [
                        ('projection', nn.Linear(self.hidden_sizes[-1], 1)),
                        ('sigmoidal', ACTIVATION_FN_FACTORY['sigmoid'])
                    ]
                )
            )
        )

    def forward(self, smiles, gep, confidence=False):
        gep = torch.unsqueeze(gep, dim=1)
        embedded_smiles = self.smiles_embedding(smiles.to(dtype=torch.int64))

        # SMILES convolutions. Unsqueeze has shape bs x 1 x T X T
        encoded_smiles = [embedded_smiles] + [
            self.convolutional_layer[ind]
            (torch.unsqueeze(embedded_smiles, 1)).permute(0, 2, 1)
            for ind in range(len(self.convolutional_layer))
        ]

        # Molecule context attention
        encodings, smiles_alphas, gene_alphas = [], [], []
        for layer in range(len(self.molecule_heads)):
            for head in range(self.molecule_heads[layer]):
                ind = self.molecule_heads[0] * layer + head
                e, a = self.molecule_attention_layers[ind](
                    encoded_smiles[layer], gep
                )
                encodings.append(e)
                smiles_alphas.append(a)

        encodings = torch.cat(encodings, dim=1)

        # Apply batch normalization if specified
        inputs = self.batch_norm(encodings) if self.params.get(
            'batch_norm', False
        ) else encodings

        # NOTE: stacking dense layers as a bottleneck
        for dl in self.dense_layers:
            inputs = dl(inputs)

        predictions = self.final_dense(inputs)
        prediction_dict = {}

        if not self.training:
            # The below is to ease postprocessing
            smiles_attention = torch.cat(
                [torch.unsqueeze(p, -1) for p in smiles_alphas], dim=-1
            )
            gene_attention = torch.cat(
                [torch.unsqueeze(p, -1) for p in gene_alphas], dim=-1
            )
            prediction_dict.update({
                'gene_attention': gene_attention,
                'smiles_attention': smiles_attention,
                'IC50': predictions,
                'log_micromolar_IC50':
                    get_log_molar(
                        predictions,
                        ic50_max=self.IC50_max,
                        ic50_min=self.IC50_min
                    ) if self.min_max_scaling else predictions
            })  # yapf: disable

        return predictions, prediction_dict
