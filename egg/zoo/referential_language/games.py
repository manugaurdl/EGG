# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn.functional as F

from egg.core.gs_wrappers import (
    GumbelSoftmaxWrapper,
    SymbolGameGS,
    SymbolReceiverWrapper,
)
from egg.core.interaction import LoggingStrategy
from egg.zoo.referential_language.archs import (
    Receiver,
    Sender,
    VisionWrapper,
    get_cnn,
)


def loss(
    _sender_input,
    _message,
    _receiver_input,
    receiver_output,
    _labels,
    aux_input,
):
    labels = aux_input["game_labels"].view(-1)
    mask = aux_input["mask"]

    acc = (receiver_output.argmax(dim=-1) == labels).detach().float()
    acc = (acc.sum() / mask.sum()).unsqueeze(0)  # avoid dimensionless tensors

    loss = F.cross_entropy(receiver_output, labels, reduction="none")
    loss = loss * mask.view(-1)  # multiply by 0 masked elements
    aux = {"acc": acc}
    if "baseline" in aux_input:
        aux["baseline"] = aux_input["baseline"].squeeze()
    return loss, aux


def build_game(opts):
    cnn_sender, input_dim = get_cnn(opts.vision_model_name, opts.pretrain_vision)
    cnn_receiver = None
    if not opts.shared_vision:
        assert not opts.pretrain_vision
        cnn_receiver, _ = get_cnn(opts.vision_model_name, opts.pretrain_vision)

    sender = Sender(
        input_dim=input_dim,
        output_dim=opts.vocab_size,
    )
    receiver = Receiver(
        input_dim=input_dim,
        output_dim=opts.output_dim,
        temperature=opts.loss_temperature,
    )
    sender = GumbelSoftmaxWrapper(
        agent=sender,
        temperature=opts.gs_temperature,
    )
    receiver = SymbolReceiverWrapper(
        receiver,
        opts.vocab_size,
        opts.output_dim,
    )
    game = SymbolGameGS(
        sender=sender,
        receiver=receiver,
        loss=loss,
        train_logging_strategy=LoggingStrategy.minimal(),
        test_logging_strategy=LoggingStrategy.minimal(),
    )
    return VisionWrapper(game, cnn_sender, cnn_receiver)
