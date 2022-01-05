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
    VisionWrapper,
    initialize_vision_module,
)


def loss(
    _sender_input,
    _message,
    _receiver_input,
    receiver_output,
    _labels,
    aux_input,
):
    labels, logits = [], []
    n_objs = receiver_output.shape[1]
    for idx, mask_elem in enumerate(aux_input["mask"]):
        similarities = receiver_output[idx]
        label = torch.arange(n_objs, device=receiver_output.device)

        pad_idxs = torch.nonzero(mask_elem)
        if pad_idxs.numel() != 0:
            # idx of first nonzero elem is when masking/padding starts
            begin_pad = pad_idxs[0].item()
            label[begin_pad:] = -100
            similarities[:, begin_pad:] = -float("inf")

        logits.append(similarities)
        labels.append(label)

    logits = torch.cat(logits)
    labels = torch.cat(labels)

    [bsz, max_objs] = aux_input["mask"].shape
    acc = (logits.argmax(dim=1) == labels).detach().float().view(bsz, max_objs)
    loss = F.cross_entropy(logits, labels, reduction="none")
    return loss, {"acc": acc, "baseline": aux_input["baselines"]}


def get_vision_modules(opts):
    vision_module_sender, input_dim = initialize_vision_module(
        name=opts.vision_model_name, pretrained=opts.pretrain_vision
    )
    vision_module_receiver = None
    if not opts.shared_vision:
        vision_module_receiver, _ = initialize_vision_module(
            name=opts.vision_model_name
        )
    return vision_module_sender, vision_module_receiver, input_dim


def get_logging_strategies():
    train_logging_strategy = LoggingStrategy.minimal()
    logging_test_args = [False, True, True, True, True, True, False]
    test_logging_strategy = LoggingStrategy(*logging_test_args)
    return train_logging_strategy, test_logging_strategy


def build_gs_game(opts):
    train_logging_strategy, test_logging_strategy = get_logging_strategies()
    vision_module_sender, vision_module_receiver, sender_input_dim = get_vision_modules(
        opts
    )
    sender = Sender(
        input_dim=sender_input_dim,
        output_dim=opts.vocab_size,
        num_heads=opts.num_heads,
        attention_type=opts.attention_type,
        context_integration=opts.context_integration,
    )
    receiver = Receiver(
        input_dim=sender_input_dim,
        hidden_dim=opts.recv_hidden_dim,
        output_dim=opts.recv_output_dim,
        temperature=opts.recv_temperature,
        use_cosine_sim=opts.use_cosine_similarity,
    )
    sender = GumbelSoftmaxWrapper(
        agent=sender,
        temperature=opts.gs_temperature,
    )
    receiver = SymbolReceiverWrapper(
        receiver,
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
    return VisionWrapper(game, vision_module_sender, vision_module_receiver)


def build_game(opts):
    game_mode2game_fn = {"gs": build_gs_game}
    assert opts.game_mode in ["gs"], f"Cannot recognize {opts.game_mode}"

    game = game_mode2game_fn[opts.game_mode](opts)
    if opts.distributed_context.is_distributed:
        return torch.nn.SyncBatchNorm.convert_sync_batchnorm(game)
    return game
