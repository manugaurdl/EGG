# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


class LinearSender(nn.Module):
    def __init__(
        self,
        input_dim: int,
        vocab_size: int = 2048,
    ):
        super(LinearSender, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, vocab_size),
            nn.BatchNorm1d(vocab_size),
        )

    def forward(self, resnet_output, aux_input=None):
        return self.fc(resnet_output)


class Receiver(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 2048,
        temperature: float = 1.0,
    ):
        super(Receiver, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )
        self.temperature = temperature

    def forward(self, message, resnet_output, aux_input=None):
        distractors = self.fc(resnet_output)
        similarity_scores = (
            torch.nn.functional.cosine_similarity(
                message.unsqueeze(1), distractors.unsqueeze(0), dim=2
            )
            / self.temperature
        )
        return similarity_scores
