# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from egg.core.continous_communication import SenderReceiverContinuousCommunication
from egg.core.gs_wrappers import GumbelSoftmaxWrapper, SymbolReceiverWrapper
from egg.core.interaction import LoggingStrategy
from egg.core.losses import DiscriminationLoss
from egg.zoo.emcom_as_ssl.informed_sender.archs import (
    InformedSender,
    ReceiverWithInformedSender,
    VisionGameWrapperWithInformed,
)
from egg.zoo.emcom_as_ssl.utils_game import build_vision_encoder


def xent_loss_with_informed(
    _sender_input, _message, _receiver_input, receiver_output, _labels, aux_input=None
):
    labels = aux_input["target_position"]
    return DiscriminationLoss.discrimination_loss(receiver_output, labels)


def xent_loss_force_compare_two(
    _sender_input, _message, _receiver_input, receiver_output, _labels, aux_input=None
):
    guess = torch.argmax(receiver_output, dim=-1)
    labels = aux_input["target_position"].view(-1, 1).repeat((1, 128))
    acc = (
        (
            torch.sum((guess == labels), 1)
            == torch.Tensor([guess.shape[-1]] * 2).to(guess.device)
        )
        .float()
        .mean()
    )
    return 0.0, {"acc": acc}


def build_game(opts):
    vision_encoder, visual_features_dim = build_vision_encoder(
        model_name=opts.model_name,
        shared_vision=opts.shared_vision,
        pretrain_vision=opts.pretrain_vision,
    )

    train_logging_strategy = LoggingStrategy(False, False, True, True, True, False)
    test_logging_strategy = LoggingStrategy(False, False, True, False, False, False)

    sender = InformedSender(
        input_dim=visual_features_dim,
        vocab_size=opts.vocab_size,
        game_size=opts.batch_size // 2,
        force_compare_two=opts.force_compare_two,
    )
    receiver = ReceiverWithInformedSender(
        input_dim=visual_features_dim,
        output_dim=opts.output_dim,
        temperature=opts.similarity_temperature,
        force_compare_two=opts.force_compare_two,
    )
    if opts.force_compare_two:
        loss = xent_loss_force_compare_two
    else:
        loss = xent_loss_with_informed

    sender = GumbelSoftmaxWrapper(agent=sender, temperature=opts.gs_temperature)
    receiver = SymbolReceiverWrapper(
        agent=receiver, vocab_size=opts.vocab_size, agent_input_size=opts.output_dim
    )
    game = SenderReceiverContinuousCommunication(
        sender,
        receiver,
        loss,
        train_logging_strategy=train_logging_strategy,
        test_logging_strategy=test_logging_strategy,
    )

    game = VisionGameWrapperWithInformed(game, vision_encoder)
    if opts.distributed_context.is_distributed:
        game = torch.nn.SyncBatchNorm.convert_sync_batchnorm(game)

    return game
