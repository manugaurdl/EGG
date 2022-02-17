# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch.nn.functional as F

from egg.core.gs_wrappers import SymbolGameGS
from egg.core.interaction import LoggingStrategy
from egg.zoo.referential_language.archs import (
    Attention_topk,
    SingleSymbolReceiverWrapper,
    DoubleSymbolReceiverWrapper,
    MessageGeneratorMLP,
    NoAttention,
    Receiver,
    ScaledDotProductAttention,
    Sender,
    SelfAttention,
    VisionWrapper,
    get_cnn,
)


def loss(
    self,
    _sender_input,
    _message,
    _receiver_input,
    receiver_output,
    _labels,
    aux_input,
):
    bsz, max_objs = aux_input["mask"].shape
    labels = aux_input["game_labels"].view(-1)
    mask = aux_input["mask"].float().view(-1)

    all_accs = (receiver_output.argmax(dim=-1) == labels).detach().float()
    aux_input["all_accs"] = all_accs.view(bsz, max_objs, -1)

    loss = F.cross_entropy(receiver_output, labels, reduction="none")
    loss *= mask  # multiply by 0 masked elements
    acc = ((all_accs * mask).sum() / mask.sum()).unsqueeze(0)
    return loss, {"acc": acc, "baseline": aux_input["baseline"]}


def build_attention(opts):
    if opts.attn_type == "top":
        attn_type = Attention_topk(
            img_feats_dim=opts.img_feats_dim, k=opts.attn_topk, random=opts.random_k
        )
    elif opts.attn_type == "dot":
        attn_type = ScaledDotProductAttention(img_feats_dim=opts.img_feats_dim)
    elif opts.attn_type == "self":
        attn_type = SelfAttention(
            embed_dim=opts.img_feats_dim, num_heads=opts.num_heads
        )
    elif opts.attn_type == "none":
        attn_type = NoAttention()
    else:
        raise NotImplementedError
    return attn_type


def build_message_generator(opts):
    return MessageGeneratorMLP(
        input_dim=opts.img_feats_dim,
        output_dim=opts.vocab_size,
        single_symbol=opts.single_symbol,
        temperature=opts.gs_temperature,
        separate_mlps=opts.sender_separate_mlps,
        random_msg_position=opts.random_msg_position,
    )


def build_sender(opts):
    return Sender(
        attn_fn=build_attention(opts), msg_generator=build_message_generator(opts)
    )


def build_receiver(opts):
    receiver = Receiver(
        input_dim=opts.img_feats_dim,
        output_dim=opts.output_dim,
        temperature=opts.loss_temperature,
    )
    args = [receiver, opts.vocab_size, opts.output_dim, opts.recv_separate_embeddings]
    if opts.single_symbol:
        return SingleSymbolReceiverWrapper(*args)
    return DoubleSymbolReceiverWrapper(*args)


def build_game(opts):
    visual_encoder = get_cnn(opts)
    sender = build_sender(opts)
    receiver = build_receiver(opts)

    game = SymbolGameGS(
        sender=sender,
        receiver=receiver,
        loss=loss,
        train_logging_strategy=LoggingStrategy.minimal(),
        test_logging_strategy=LoggingStrategy.minimal(),
    )
    if opts.distributed_context.is_distributed:
        game = nn.SyncBatchNorm.convert_sync_batchnorm(game)
    return VisionWrapper(game, visual_encoder)
