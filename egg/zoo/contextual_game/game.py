# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

from egg.core.gs_wrappers import (
    GumbelSoftmaxWrapper,
    SymbolReceiverWrapper,
    SymbolGameGS,
)
from egg.zoo.contextual_game.archs import (
    Receiver,
    Sender,
    VisionWrapper,
    initialize_vision_module,
)


def loss(
    _sender_input,
    _message,
    _receiver_input,
    receiver_output,
    labels,
    _aux_input,
):
    captioned_img = receiver_output[labels]
    captioned_img_acc = (captioned_img.argmax(dim=1) == labels).detach().float()

    all_img_labels = torch.arange(receiver_output.shape[0]).to(receiver_output.device)
    all_imgs_acc = (receiver_output.argmax(dim=1) == all_img_labels).detach().float()

    loss = F.cross_entropy(captioned_img, labels, reduction="none")
    return loss, {"acc": captioned_img_acc, "all_accs": all_imgs_acc}


def build_game(opts):
    vision_model, visual_feats_size = initialize_vision_module(
        opts.vision_model, opts.pretrain_vision
    )

    sender = Sender(
        input_dim=visual_feats_size,
        vocab_size=opts.vocab_size,
    )
    sender = GumbelSoftmaxWrapper(sender, temperature=opts.gs_temperature)

    receiver = Receiver(
        input_dim=visual_feats_size,
        hidden_dim=opts.recv_hidden_dim,
        output_dim=opts.recv_output_dim,
        temperature=opts.recv_temperature,
    )
    receiver = SymbolReceiverWrapper(
        receiver,
        opts.vocab_size,
        opts.recv_output_dim,
    )

    game = SymbolGameGS(sender=sender, receiver=receiver, loss=loss)
    if opts.distributed_context.is_distributed:
        game = torch.nn.SyncBatchNorm.convert_sync_batchnorm(game)

    return VisionWrapper(game, vision_model)
