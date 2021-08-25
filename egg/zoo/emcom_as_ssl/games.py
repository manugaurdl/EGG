# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

from egg.core.continous_communication import SenderReceiverContinuousCommunication
from egg.core.gs_wrappers import GumbelSoftmaxWrapper, SymbolReceiverWrapper
from egg.core.interaction import LoggingStrategy
from egg.zoo.emcom_as_ssl.archs import (
    EmSSLSender,
    Receiver,
    VisionGameWrapper,
    VisionModule,
    get_vision_modules,
)


def build_vision_encoder(
    model_name: str = "resnet50",
    shared_vision: bool = False,
    pretrain_vision: bool = False,
):
    (
        sender_vision_module,
        receiver_vision_module,
        visual_features_dim,
    ) = get_vision_modules(
        encoder_arch=model_name, shared=shared_vision, pretrain_vision=pretrain_vision
    )
    vision_encoder = VisionModule(
        sender_vision_module=sender_vision_module,
        receiver_vision_module=receiver_vision_module,
    )
    return vision_encoder, visual_features_dim


def xent_loss(_sender_input, _message, _receiver_input, receiver_output, _labels):
    batch_size = receiver_output.shape[0]

    labels = torch.arange(batch_size, device=receiver_output.device)
    acc = (receiver_output.argmax(dim=1) == labels).detach().float()
    loss = F.cross_entropy(receiver_output, labels, reduction="none")
    return loss, {"acc": acc}


def build_game(opts):
    vision_encoder, visual_features_dim = build_vision_encoder(
        model_name=opts.model_name,
        shared_vision=opts.shared_vision,
        pretrain_vision=opts.pretrain_vision,
    )

    train_logging_strategy = LoggingStrategy(False, False, True, True, True, False)
    test_logging_strategy = LoggingStrategy(False, False, True, False, False, False)

    sender = GumbelSoftmaxWrapper(
        EmSSLSender(
            input_dim=visual_features_dim,
            vocab_size=opts.vocab_size,
        ),
        temperature=opts.gs_temperature,
    )
    receiver = SymbolReceiverWrapper(
        Receiver(
            input_dim=visual_features_dim,
            output_dim=opts.output_dim,
            temperature=opts.similarity_temperature,
        ),
        opts.vocab_size,
        opts.output_dim,
    )

    game = SenderReceiverContinuousCommunication(
        sender,
        receiver,
        xent_loss,
        train_logging_strategy=train_logging_strategy,
        test_logging_strategy=test_logging_strategy,
    )

    game = VisionGameWrapper(game, vision_encoder)
    if opts.distributed_context.is_distributed:
        game = torch.nn.SyncBatchNorm.convert_sync_batchnorm(game)

    return game
