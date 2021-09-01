# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


class VisionGameWrapperWithInformed(nn.Module):
    def __init__(
        self,
        game: nn.Module,
        vision_module: nn.Module,
    ):
        super(VisionGameWrapperWithInformed, self).__init__()
        self.game = game
        self.vision_module = vision_module

    def forward(self, sender_input, labels, receiver_input=None, aux_input=None):
        sender_encoded_input, receiver_encoded_input = self.vision_module(
            sender_input, receiver_input
        )
        bsz = sender_encoded_input.shape[0]
        sender_encoded_input = sender_encoded_input.view(2, 1, bsz // 2, -1)
        receiver_encoded_input = receiver_encoded_input.view(2, 1, bsz // 2, -1)
        random_order = aux_input["random_order"]
        receiver_encoded_input1 = receiver_encoded_input[0, 0, random_order[0]]
        receiver_encoded_input2 = receiver_encoded_input[1, 0, random_order[1]]
        receiver_encoded_input = torch.stack(
            [receiver_encoded_input1, receiver_encoded_input2]
        )

        return self.game(
            sender_input=sender_encoded_input,
            labels=labels,
            receiver_input=receiver_encoded_input,
            aux_input=aux_input,
        )


class InformedSender(nn.Module):
    def __init__(
        self,
        input_dim: int,  # feat_size,
        hidden_dim: int = 20,
        embedding_dim: int = 50,
        vocab_size: int = 2048,
        game_size: int = 2,  # distractors + 1 target)
        force_compare_two: bool = False,
    ):
        super(InformedSender, self).__init__()

        self.force_compare_two = force_compare_two

        # testing a model trained with 1 distractors on 127 distractors by computing 64
        if self.force_compare_two:
            game_size = 2

        self.fc_in = nn.Linear(input_dim, embedding_dim, bias=False)
        self.conv1 = nn.Conv2d(
            1,
            hidden_dim,
            kernel_size=(game_size, 1),
            stride=(game_size, 1),
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            1, 1, kernel_size=(hidden_dim, 1), stride=(hidden_dim, 1), bias=False
        )
        self.lin2 = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.fc_out = nn.Linear(vocab_size, embedding_dim, bias=False)

    def compute_message(self, emb):
        # in: h of size (batch_size, 1, game_size, embedding_size)
        # out: h of size (batch_size, hidden_size, 1, embedding_size)
        h = self.conv1(emb)
        h = torch.sigmoid(h)
        # in: h of size (batch_size, hidden_size, 1, embedding_size)
        # out: h of size (batch_size, 1, hidden_size, embedding_size)
        h = h.transpose(1, 2)
        h = self.conv2(h)
        # h of size (batch_size, 1, 1, embedding_size)
        h = torch.sigmoid(h)
        # h of size (batch_size, embedding_size)
        h = self.lin2(h)
        h = h.squeeze(1).squeeze(1)
        # h of size (batch_size, vocab_size)
        return h

    def forward(self, x, _aux_input=None):
        bsz = x.shape[0]
        emb = self.fc_in(x)
        if (not self.training) and self.force_compare_two:
            msgs = []
            for batch_idx in range(bsz):
                for distractor_idx in range(emb.shape[2] - 1):
                    candidate_pair = (
                        torch.stack(
                            [
                                emb[batch_idx, 0, 0],
                                emb[batch_idx, 0, distractor_idx + 1],
                            ]
                        )
                        .unsqueeze(0)
                        .unsqueeze(0)
                    )
                    msgs.append(self.compute_message(candidate_pair))
            result = torch.cat(msgs).view(bsz, emb.shape[2] - 1, -1)
            return result
        return self.compute_message(emb)


class ReceiverWithInformedSender(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 2048,
        temperature: float = 1.0,
        force_compare_two: bool = False,
    ):
        super(ReceiverWithInformedSender, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.temperature = temperature
        self.force_compare_two = force_compare_two

    def forward(self, message, resnet_output, aux_input=None):
        distractors = self.fc(resnet_output)
        if (not self.training) and self.force_compare_two:
            bsz = message.shape[0]
            message = torch.cat(
                [
                    message,
                    torch.zeros(bsz, 1, message.shape[-1], device=message.device),
                ],
                dim=1,
            )
            similarity_scores = (
                torch.nn.functional.cosine_similarity(
                    message.unsqueeze(2), distractors.unsqueeze(1), dim=3
                )
                / self.temperature
            )
            return similarity_scores
        else:
            similarity_scores = (
                torch.nn.functional.cosine_similarity(
                    message.unsqueeze(1), distractors, dim=2
                )
                / self.temperature
            )
            return similarity_scores
