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
        output_dim,
        temperature,
    ):
        super(CatMLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LeakyReLU(),
            nn.Linear(output_dim, output_dim),
        )

        self.temperature = temperature

    def forward(self, tgt_embedding, ctx_embedding, aux_input=None):
        bsz, max_objs, _ = tgt_embedding.shape

        x = torch.cat([tgt_embedding, ctx_embedding], dim=-1)

        logits = self.fc(x.view(bsz * max_objs, -1))

        message = gumbel_softmax_sample(logits, self.temperature, self.training)

        return message
