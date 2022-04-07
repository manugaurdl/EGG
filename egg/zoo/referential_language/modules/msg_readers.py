# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


class MultipleSymbolReader(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, output_dim):
        super(MultipleSymbolReader, self).__init__()
        self.fc_in = nn.Sequential(
            nn.Linear(vocab_size, embedding_dim),
            nn.LeakyReLU(),
        )
        self.fc_out = nn.Linear(2 * embedding_dim, output_dim)

    def forward(self, messages, aux_input=None):
        bsz, max_objs, num_symbols, _ = messages.shape
        assert num_symbols == 2
        messages = messages.view(bsz * max_objs, num_symbols, -1)

        embedding_sym1 = self.fc_in(messages[..., 0, :])
        embedding_sym2 = self.fc_in(messages[..., 1, :])

        msg_embedding = torch.cat([embedding_sym1, embedding_sym2], dim=1)
        return self.fc_out(msg_embedding)
