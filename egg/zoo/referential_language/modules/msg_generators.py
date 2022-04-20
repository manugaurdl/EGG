# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from egg.core.gs_wrappers import gumbel_softmax_sample as gumbel_sample


class CatMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        vocab_size,
        gs_temperature,
        **kwargs,
    ):
        super(CatMLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, vocab_size),
            nn.LeakyReLU(),
            nn.Linear(vocab_size, vocab_size),
        )

        self.temperature = gs_temperature

    def forward(self, tgt_embedding, ctx_embedding, aux_input=None):
        bsz, max_objs, _ = tgt_embedding.shape

        x = torch.cat([tgt_embedding, ctx_embedding], dim=-1)

        logits = self.fc(x.view(bsz * max_objs, -1))

        message = gumbel_sample(logits, self.temperature, self.training)

        return message


class ConditionalMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        embedding_dim,
        vocab_size,
        gs_temperature,
        num_layers=1,
        shared_mlp=True,
        context_first=False,
        activation="leaky_relu",
        **kwargs,
    ):
        super(ConditionalMLP, self).__init__()
        activations = {
            "relu": F.relu,
            "tanh": F.tanh,
            "leaky_relu": F.leaky_relu,
            "identity": nn.Identity(),
        }
        self.activation = activations[activation.lower()]

        assert num_layers > 0
        encoder_hidden_sizes = [embedding_dim] * num_layers
        encoder_layer_dimensions = [(input_dim, encoder_hidden_sizes[0])]

        for i, hidden_size in enumerate(encoder_hidden_sizes[1:]):
            hidden_shape = (encoder_hidden_sizes[i], hidden_size)
            encoder_layer_dimensions.append(hidden_shape)

        self.fc_tgt = nn.ModuleList(
            [nn.Linear(*dimensions) for dimensions in encoder_layer_dimensions]
        )
        if shared_mlp:
            self.fc_ctx = self.fc_tgt
        else:
            self.fc_ctx = nn.ModuleList(
                [nn.Linear(*dimensions) for dimensions in encoder_layer_dimensions]
            )

        self.embedding_to_vocab = nn.Linear(embedding_dim, vocab_size)
        self.target_to_embedding = nn.Linear(vocab_size, embedding_dim)

        self.context_first = context_first

        self.fc_out = nn.Linear(embedding_dim * 2, vocab_size)

        self.temperature = gs_temperature

    def forward(self, tgt, ctx, aux_input=None):
        for hidden_layer in self.fc_tgt[:-1]:
            tgt = self.activation(hidden_layer(tgt))
        tgt_embedding = self.fc_tgt[-1](tgt)

        for hidden_layer in self.fc_ctx[:-1]:
            ctx = self.activation(hidden_layer(ctx))
        ctx_embedding = self.fc_ctx[-1](ctx)

        first_embedding = ctx_embedding if self.context_first else tgt_embedding
        second_embedding = tgt_embedding if self.context_first else ctx_embedding

        logits_sym1 = self.embedding_to_vocab(first_embedding)

        symbol1 = gumbel_sample(logits_sym1, self.temperature, self.training)

        symbol1_embedding = self.target_to_embedding(symbol1)

        contextualized_logits = self.fc_out(
            torch.cat([symbol1_embedding, second_embedding], dim=-1)
        )

        contextualized_symbol = gumbel_sample(
            contextualized_logits, self.temperature, self.training
        )
        return torch.stack([symbol1, contextualized_symbol], dim=2)
