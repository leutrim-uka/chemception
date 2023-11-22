import torch
from torch import nn

from utils import embed_tokens


# dense attention layer
# embedding layer
# sequence attention layer
# contextual attention layer
# contextual attention matrix layer


class MCA:
    activation_fn_map = {
        'relu': nn.functional.relu,
        'sigmoid': nn.functional.sigmoid,
        'tanh': nn.functional.tanh,
        'elu': nn.functional.elu,
        'selu': nn.functional.selu,
        'leaky_relu': nn.functional.leaky_relu,
        'softmax': nn.functional.softmax
    }

    default_params = {
        "batch_size": 512,
        "learning_rate": 0.0002,
        "dropout": 0.3,
        "batch_norm": True,
        "loss_function": "mse",
        "stacked_dense_hidden_sizes": [512, 128, 64, 16],
        "activation": "relu",
        "selected_genes_name": "selected_genes_20",
        "tokens_name": "smiles_atom_tokens",
        "smiles_vocabulary_size": 28,
        "smiles_embedding_size": 16,
        "multiheads": [4, 4, 4, 4],
        "filters": [64, 64, 64],
        "kernel_sizes": [[3, 16], [5, 16], [11, 16]],
        "smiles_attention": "contextual",
        "smiles_attention_size": 64,
        "dense_attention": True,
        "gene_multihead": True,
        "buffer_size": 1000000,
        "prefetch_buffer_size": 512,
        "number_of_threads": 10,
        "drop_remainder": True,
        "eval_batch_size": 32
    }

    def __init__(self, features, labels, mode, params):
        super(MCA, self).__init__()
        self.features = features
        self.labels = labels
        self.mode = mode
        self.params = params

    def train(self):
        dropout = self.params.get('dropout', 0.0)
        batch_size = self.params.get('batch_size')
        tokens = self.features['tokens']
        sequence_length = tokens.shape[1].value

        # get the 2128 selected genes - 20 per drug - transcriptomic profile
        genes = self.params.get('selected_genes_name')

        num_gene_features = 1 if len(genes.shape) == 2 else genes.shape[2].value

        activation_fn = MCA.activation_fn_map[self.params.get('activation', 'relu')]

        embedded_tokens = embed_tokens(
            tokens, self.params['smiles_vocabulary_size'],
            self.params['smiles_embedding_size'],
            # name='smiles_embedding'
        )

        filters = self.params.get('filters')
        kernel_sizes = self.params.get(
            'kernel_sizes',
            [
                [3, self.params['smiles_embedding_size']],
                [5, self.params['smiles_embedding_size']]
            ]
        )
        multiheads = self.params.get('multiheads')

        assert len(filters) == len(kernel_sizes)
        assert len(filters) + 1 == len(multiheads)

    def attention_list_to_matrix(self, coding_tuple, axis=2):
        pass
