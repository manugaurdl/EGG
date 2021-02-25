# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from egg.core.continous_communication import (
    ContinuousLinearSender,
    SenderReceiverContinuousCommunication
)
from egg.core.interaction import LoggingStrategy
from egg.core.reinforce_wrappers import (
    RnnSenderReinforce,
    SenderReceiverRnnReinforce,
)
from egg.zoo.simclr.archs import (
    Receiver,
    RnnReceiverDeterministicContrastive,
    VisionGameWrapper,
    VisionModule
)
from egg.zoo.simclr.losses import Loss


def build_game(opts):
    device = torch.device("cuda" if opts.cuda else "cpu")
    vision_encoder = VisionModule(
        encoder_arch=opts.model_name,
        projection_dim=opts.vision_projection_dim,
        shared=opts.shared_vision
    )

    train_logging_strategy = LoggingStrategy.minimal()
    loss = Loss(opts.batch_size, opts.ntxent_tau, device)

    if opts.communication_channel == "rf":
        sender = RnnSenderReinforce(
            agent=nn.Identity(),
            vocab_size=opts.vocab_size,
            embed_dim=opts.sender_embedding,
            hidden_size=opts.vision_projection_dim,
            max_len=opts.max_len,
            num_layers=opts.sender_rnn_num_layers,
            cell=opts.sender_cell
        )
        receiver = Receiver(
            msg_input_dim=opts.receiver_rnn_hidden,
            img_feats_input_dim=opts.vision_projection_dim,
            output_dim=opts.receiver_output_size
        )
        receiver = RnnReceiverDeterministicContrastive(
            receiver,
            vocab_size=opts.vocab_size,
            embed_dim=opts.receiver_embedding,
            hidden_size=opts.receiver_rnn_hidden,
            cell=opts.receiver_cell,
            num_layers=opts.receiver_num_layers
        )
        game = SenderReceiverRnnReinforce(
            sender,
            receiver,
            loss,
            sender_entropy_coeff=opts.sender_entropy_coeff,
            train_logging_strategy=train_logging_strategy
        )
    elif opts.communication_channel == "continuous":
        sender = ContinuousLinearSender(
            agent=nn.Identity(),
            encoder_input_size=opts.vision_projection_dim,
            encoder_hidden_size=opts.sender_output_size
        )
        receiver = Receiver(
            msg_input_dim=opts.sender_output_size,
            img_feats_input_dim=opts.vision_projection_dim,
            output_dim=opts.receiver_output_size
        )

        game = SenderReceiverContinuousCommunication(
            sender,
            receiver,
            loss,
            train_logging_strategy
        )

    game = VisionGameWrapper(game, vision_encoder)
    return game
