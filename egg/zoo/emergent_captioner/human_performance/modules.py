# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable

import torch
import torch.nn as nn

from egg.core.interaction import LoggingStrategy


class ZeroShotCaptionGame(nn.Module):
    def __init__(
        self,
        sender: nn.Module,
        receiver: nn.Module,
        loss: Callable,
        logging_strategy: LoggingStrategy = None,
    ):
        super(ZeroShotCaptionGame, self).__init__()

        self.train_logging_strategy = LoggingStrategy.minimal()
        self.test_logging_strategy = (
            LoggingStrategy.minimal() if logging_strategy is None else logging_strategy
        )

        self.sender = sender
        self.receiver = receiver
        self.loss = loss

    def forward(self, sender_input, labels, receiver_input=None, aux_input=None):
        message = self.sender(sender_input, aux_input)
        message_length = torch.Tensor([len(x) for x in message]).int()

        receiver_output = self.receiver(message, receiver_input, aux_input)

        loss, aux_info = self.loss(
            sender_input, message, receiver_input, receiver_output, labels, aux_input
        )

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )

        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input,
            labels=labels,
            receiver_input=receiver_input,
            aux_input=aux_input,
            message=message,
            receiver_output=receiver_output.detach(),
            message_length=message_length,
            aux=aux_info,
        )
        return loss.mean(), interaction


class HumanCaptionSender(nn.Module):
    def forward(self, x, aux_input=None):
        return aux_input["caption"]
