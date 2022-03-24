# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
import torch.nn as nn

from egg.core.interaction import LoggingStrategy


class Sender(nn.Module):
    def __init__(
        self,
        attn_fn: nn.Module,
        msg_generator: nn.Module,
    ):
        super(Sender, self).__init__()
        self.attn_fn = attn_fn
        self.msg_generator = msg_generator

    def forward(self, x, aux_input=None):
        attn = self.attn_fn(x, aux_input)
        return self.msg_generator(x, attn, aux_input)


class Receiver(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        temperature: int,
        msg_reader: nn.Module,
    ):
        super(Receiver, self).__init__()
        self.fc_img = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LeakyReLU(),
            nn.Linear(output_dim, output_dim),
        )
        self.temperature = temperature
        self.msg_reader = msg_reader

    def forward(self, messages, images, aux_input=None):
        bsz, max_objs, _ = images.shape

        images = self.fc_img(images.view(bsz * max_objs, -1))
        images = images.view(bsz, max_objs, -1)
        messages = self.msg_reader(messages).view(bsz, max_objs, -1)

        sims = torch.bmm(messages, images.transpose(-1, -2)) / self.temperature
        return sims.view(bsz * max_objs, -1)


class VisionWrapper(nn.Module):
    def __init__(
        self,
        game: nn.Module,
        visual_encoder: nn.Module,
        train_logging_strategy: Optional[LoggingStrategy] = None,
        test_logging_strategy: Optional[LoggingStrategy] = None,
    ):
        super(VisionWrapper, self).__init__()
        self.game = game
        self.visual_encoder = visual_encoder

        self.train_logging_strategy = (
            LoggingStrategy().minimal()
            if train_logging_strategy is None
            else train_logging_strategy
        )
        self.test_logging_strategy = (
            LoggingStrategy().minimal()
            if test_logging_strategy is None
            else test_logging_strategy
        )

    def forward(self, sender_input, labels, receiver_input=None, aux_input=None):
        bsz, max_objs, _, h, w = sender_input.shape

        sender_feats = self.visual_encoder(sender_input.view(bsz * max_objs, 3, h, w))
        sender_input = sender_feats.view(bsz, max_objs, -1)
        recv_input = sender_input

        # recv_feats = self.visual_encoder(receiver_input.view(bsz * max_objs, 3, h, w))
        # recv_input = recv_feats.view(bsz, max_objs, -1)

        if not self.training:
            aux_input["sender_img_feats"] = sender_input
            # aux_input["recv_img_feats"] = recv_input

        loss, interaction = self.game(sender_input, labels, recv_input, aux_input)

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )

        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input,
            receiver_input=receiver_input,
            labels=labels,
            aux_input=interaction.aux_input,
            receiver_output=interaction.receiver_output,
            message=interaction.message,
            message_length=interaction.message_length,
            aux=interaction.aux,
        )
        return loss, interaction
