# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from egg.core.gs_wrappers import gumbel_softmax_sample


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

        message = gumbel_softmax_sample(logits, self.temperature, self.training)

        return message


class ConditionalMLP(nn.Module):
    def __init__(self, input_dim, embedding_dim, vocab_size, gs_temperature, **kwargs):
        super(ConditionalMLP, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.LeakyReLU(),
        )
        self.embedding_to_vocab = nn.Linear(embedding_dim, vocab_size)
        self.target_to_embedding = nn.Linear(vocab_size, embedding_dim)

        self.fc_out = nn.Linear(embedding_dim * 2, vocab_size)

        self.temperature = gs_temperature

    def forward(self, tgt, ctx, aux_input=None):
        tgt_embedding = self.fc(tgt)
        ctx_embedding = self.fc(ctx)

        logits_tgt = self.embedding_to_vocab(tgt_embedding)

        symbol_target = gumbel_softmax_sample(
            logits_tgt, self.temperature, self.training
        )

        symbol_target_embedding = self.target_to_embedding(symbol_target)

        contextualized_logits = self.fc_out(
            torch.cat([symbol_target_embedding, ctx_embedding], dim=-1)
        )

        contextualized_symbol = gumbel_softmax_sample(
            contextualized_logits, self.temperature, self.training
        )
        return torch.stack([symbol_target, contextualized_symbol], dim=2)
