import torch
from torch import nn
import torch.nn.functional as F

from layers.ContextualAttentionLayer import ContextualAttentionLayer
from layers.DenseAttentionLayer import DenseAttentionLayer
from layers.EmbeddingLayer import EmbeddingLayer


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
        #TODO: FIX THE IS_TRAINING MODE THING
        is_training = True

        dropout = self.params.get('dropout', 0.0)
        batch_size = self.params.get('batch_size')
        tokens = self.features['tokens']
        sequence_length = tokens.shape[1].value

        # get the 2128 selected genes - 20 per drug - transcriptomic profile
        genes = self.params.get('selected_genes_name')

        num_gene_features = 1 if len(genes.shape) == 2 else genes.shape[2].value

        activation_fn = MCA.activation_fn_map[self.params.get('activation', 'relu')]

        min_ic50 = self.params.get('min', 0.0)
        max_ic50 = self.params.get('max', 0.0)

        embedding_layer = EmbeddingLayer(
            self.params['smiles_vocabulary_size'],
            self.params['smiles_embedding_size'])

        embedded_tokens = embedding_layer(tokens)

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

        # Filter genes differently for each SMILES kernel size
        dense_attention_layer = DenseAttentionLayer()
        gene_tuple = [dense_attention_layer(genes) for head in range(len(multiheads))]

        encoded_genes = [tpl[0] for tpl in gene_tuple]
        gene_attention_coefficients_multi, gene_attention_coefficients = (
            self.attention_list_to_matrix(gene_tuple, axis=2)
        )

        #
        inputs = torch.unsqueeze(embedded_tokens, dim=3)

        # convolve through smiles here

        # TODO: implement SMILES convolution NN
        convolved_smiles = []
        """
        this is a list of tensors. Each element in the list corresponds to the result of
        a convolutional layer operation on the input data
        convolved_smiles = [
            F.batch_norm(
                F.dropout(
                    torch.squeeze(
                        F.conv2d(
                            input=self.pad_sequence(inputs, kernel_size),

                            padding=
                        )
                    )
                )
            ) for index, (num_kernel, kernel_size) in enumerate(
                zip(filters, kernel_sizes)
            )
        ]
        """

        # insert embedded tokens into the first position of the list
        convolved_smiles.insert(0, embedded_tokens)

        contextual_attention_layer = ContextualAttentionLayer(self.params.get('smiles_attention_size', 256))
        encoding_coefficient_tuple = [
            contextual_attention_layer(
                encoded_genes[layer],
                convolved_smiles[layer]
            ) for layer in range(len(convolved_smiles))
            for _ in range(multiheads[layer])
        ]

        # TODO: From now on, we concatenate genes and SMILES
        attention_coefficients_raw, attention_coefficients = (
            self.attention_list_to_matrix(
                encoding_coefficient_tuple, axis=2
            )
        )
        encoded_smiles_list = [t[0] for t in encoding_coefficient_tuple]
        encoded_smiles = [
            torch.reshape(
                encoded_smiles_list[layer],
                [-1, sequence_length*filters[layer-1]]
            ) for layer in range(1, len(encoded_smiles_list))
        ]
        encoded_smiles.insert(0, torch.reshape(
            encoded_smiles_list[0],
            [-1, sequence_length*self.params['smiles_embedding_size']]
        ))
        encoded_smiles = torch.cat(
            encoded_smiles, dim=1
        )
        encoded_smiles = encoded_smiles.view(
            batch_size,
            sequence_length * (
                self.params['smiles_embedding_size']*multiheads[0] +
                sum([a * b for a, b in zip(multiheads[1:])])
            )
        )

        # TODO: BATCH NORMALIZATION
        batch_normalization = nn.BatchNorm1d(encoded_smiles.size()[1])
        layer = batch_normalization(encoded_smiles)

        for index, dense_hidden_size in enumerate(
            self.params.get('stacked_dense_hidden_sizes', [])
        ):
            layer = nn.Linear(layer.size(-1), dense_hidden_size)
            layer = activation_fn(layer)
            layer = nn.BatchNorm1d(layer.size(-1))(layer)
            layer = F.dropout(layer, p=dropout, training=is_training)

        predictions = torch.squeeze(
            nn.Linear(layer.size(-1), 1)(layer)
        )

        prediction_dict = {
            'gene_attention': gene_attention_coefficients,
            'smiles_attention': attention_coefficients,
            'smiles_attention_raw': attention_coefficients_raw,
            'features': encoded_smiles
        }

        prediction_dict.update({
            'IC50': predictions,
            'IC50_micromolar': torch.exp(
                predictions * (max_ic50 - min_ic50) + min_ic50
            ),
        })

        prediction_dict.update(
            {'gene_attention_raw': gene_attention_coefficients_multi}
        )

        return predictions, prediction_dict


    def attention_list_to_matrix(self, coding_tuple, axis=2):
        return None, None

    #
    def pad_sequence(self, data, kernel_size) -> torch.Tensor:
        """
        Pad the sequence to match the shape of SMILES of different lengths.
        :param torch.Tensor data: a tensor
        :param  kernel_size: size of kernel applied to the sequence
        :return:
        """

        pad = torch.unsqueeze(
            embed_tokens(
                torch.zeros([self.params.get('batch_size'), 1], dtype=torch.int32),
                self.params['smiles_vocabulary_size'],
                self.params['smiles_embedding_size']
            ), dim=3
        )

        pad_size = kernel_size[0] // 2
        return torch.concat([pad] * pad_size + [data] + [pad] * pad_size, dim=1)
