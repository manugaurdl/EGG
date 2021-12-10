# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
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
    initialize_vision_module,
)


def loss(
    _sender_input,
    message,
    _receiver_input,
    receiver_output,
    _labels,
    aux_input,
):
    labels, logits = [], []
    for idx, mask_elem in enumerate(aux_input["mask"]):
        # idx of first nonzero elem is when masking/padding starts
        pad_idxs = torch.nonzero(mask_elem)
        unmasked_similarities = receiver_output[idx]
        label_size = receiver_output.shape[1]

        if pad_idxs.numel() != 0:
            begin_pad = pad_idxs[0].item()
            unmasked_similarities = receiver_output[idx][:begin_pad]
            unmasked_similarities[:, begin_pad:] = -float("inf")

        label_size = unmasked_similarities.shape[0]
        logits.append(unmasked_similarities)
        labels.append(torch.arange(label_size, device=receiver_output.device))

    logits = torch.cat(logits)
    labels = torch.cat(labels)

    acc = (logits.argmax(dim=1) == labels).detach().float()
    loss = F.cross_entropy(logits, labels, reduction="none")
    return loss, {"acc": acc, "baseline": aux_input["baselines"]}


def build_game(opts):
    train_logging_strategy = LoggingStrategy.minimal()
    logging_test_args = [False, False, True, True, True, True, False]
    test_logging_strategy = LoggingStrategy(*logging_test_args)

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
            attention=opts.attention,
            num_heads=opts.num_heads,
            context_integration=opts.context_integration,
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
            use_cosine_sim=opts.cosine_similarity,
        ),
        opts.vocab_size,
        opts.recv_output_dim,
    )

    game = SymbolGameGS(
        sender=sender,
        receiver=receiver,
        loss=loss,
        train_logging_strategy=train_logging_strategy,
        test_logging_strategy=test_logging_strategy,
    )

    if opts.distributed_context.is_distributed:
        game = torch.nn.SyncBatchNorm.convert_sync_batchnorm(game)

    return game
