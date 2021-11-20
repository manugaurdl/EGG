# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

from egg.core.gs_wrappers import GumbelSoftmaxWrapper, SymbolReceiverWrapper
from egg.core.interaction import LoggingStrategy
from egg.zoo.referential_language.archs import (
    ContextualCommunicationGame,
    Receiver,
    Sender,
    initialize_vision_module,
)


def loss(
    _sender_input,
    _message,
    _receiver_input,
    receiver_output,
    _labels,
    _aux_input,
):

    """
    def masked_softmax(x, mask, **kwargs):
        x_masked = x.clone()
        x_masked[mask == 0] = -float("inf")

        return torch.softmax(x_masked, **kwargs)
    """
    labels = torch.arange(receiver_output.shape[0], device=receiver_output.device)
    acc = (receiver_output.argmax(dim=1) == labels).detach().float()
    loss = F.cross_entropy(receiver_output, labels, reduction="none")
    return loss, {"acc": acc}


def build_game(opts):

    train_logging_strategy = LoggingStrategy.minimal()
    test_logging_strategy = LoggingStrategy.minimal()

    vision_module_sender, input_dim = initialize_vision_module(
        name=opts.vision_model_name, pretrained=opts.pretrain_vision
    )
    vision_module_receiver = vision_module_sender
    if not opts.shared_vision:
        vision_module_receiver, _ = initialize_vision_module(
            name=opts.vision_model_name
        )

    sender = GumbelSoftmaxWrapper(
        Sender(
            vision_module=vision_module_sender,
            input_dim=input_dim,
            vocab_size=opts.vocab_size,
        ),
        temperature=opts.gs_temperature,
    )
    receiver = SymbolReceiverWrapper(
        Receiver(
            vision_module=vision_module_receiver,
            input_dim=input_dim,
            hidden_dim=opts.recv_hidden_dim,
            output_dim=opts.recv_output_dim,
            temperature=opts.recv_temperature,
        ),
        opts.vocab_size,
        opts.recv_output_dim,
    )

    game = ContextualCommunicationGame(
        sender=sender,
        receiver=receiver,
        loss=loss,
        train_logging_strategy=train_logging_strategy,
        test_logging_strategy=test_logging_strategy,
    )

    if opts.distributed_context.is_distributed:
        game = torch.nn.SyncBatchNorm.convert_sync_batchnorm(game)

    return game
