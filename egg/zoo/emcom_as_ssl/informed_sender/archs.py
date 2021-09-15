# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F


class InformedSender(nn.Module):
    def __init__(
        self,
        input_dim: int,  # feat_size,
        hidden_dim: int = 20,
        embedding_dim: int = 50,
        vocab_size: int = 2048,
        game_size: int = 2,  # distractors + 1 target
        force_compare_two: bool = False,
    ):
        super(InformedSender, self).__init__()

        self.game_size = game_size
        self.force_compare_two = force_compare_two
        if force_compare_two:
            self.game_size = 2

        self.fc_in = nn.Linear(input_dim, embedding_dim, bias=False)
        self.conv1 = nn.Conv2d(
            1,
            hidden_dim,
            kernel_size=(self.game_size, 1),
            stride=(self.game_size, 1),
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
        bsz = x.shape[0] // self.game_size
        emb = self.fc_in(x)

        if (not self.training) and self.force_compare_two:
            msgs = []
            eval_bsz = x.shape[0] // 128
            emb = emb.view(eval_bsz, 1, 128, -1)
            for batch_idx in range(eval_bsz):
                for distractor_idx in range(1, emb.shape[2]):
                    candidate_pair = (
                        torch.stack(
                            [
                                emb[batch_idx, 0, 0],
                                emb[batch_idx, 0, distractor_idx],
                            ]
                        )
                        .unsqueeze(0)
                        .unsqueeze(0)
                    )
                    msgs.append(self.compute_message(candidate_pair))
            return torch.cat(msgs).view(eval_bsz, emb.shape[2] - 1, -1)

        emb = emb.view(bsz, 1, self.game_size, -1)
        return self.compute_message(emb)


class Receiver(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 2048,
        temperature: float = 1.0,
        game_size: int = 2,
        force_compare_two: bool = False,
    ):
        super(Receiver, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.temperature = temperature
        self.game_size = game_size
        self.force_compare_two = force_compare_two

    def forward(self, message, resnet_output, aux_input=None):
        distractors = self.fc(resnet_output)

        if self.force_compare_two:
            eval_bsz = distractors.shape[0] // 128
            distractors = distractors.view(eval_bsz, 128, -1)

            accs = []
            for batch_idx in range(eval_bsz):
                for distractor_idx in range(127):
                    candidate_msg = message[batch_idx, distractor_idx].unsqueeze(0)
                    candidate_imgs = torch.stack(
                        [
                            distractors[batch_idx, 0],
                            distractors[batch_idx, distractor_idx],
                        ]
                    )
                    sim = F.cosine_similarity(candidate_msg, candidate_imgs, dim=1)
                    if torch.argmax(sim).item() == 1:
                        accs.append(torch.Tensor([0]))
                        break
                accs.append(torch.Tensor([1]))
            return torch.cat(accs)

        bsz = resnet_output.shape[0] // self.game_size
        random_order = aux_input["random_order"]
        distractors = distractors.view(bsz, 1, self.game_size, -1)
        distractors = torch.stack(
            [
                distractors[batch_idx, 0, random_order[batch_idx]]
                for batch_idx in range(bsz)
            ]
        )

        similarity_scores = F.cosine_similarity(
            message.unsqueeze(1), distractors, dim=2
        )
        return similarity_scores / self.temperature
