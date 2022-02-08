# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from egg.core.gs_wrappers import SymbolGameGS
from egg.core.interaction import LoggingStrategy
from egg.zoo.referential_language.archs import (
    Attention_topk,
    SingleSymbolReceiverWrapper,
    DoubleSymbolReceiverWrapper,
    Receiver,
    ScaledDotProductAttention,
    Sender,
    SelfAttention,
    VisionWrapper,
    get_cnn,
    no_attention,
)


class Loss(nn.Module):
    def __init__(self, random_distractors: bool = False):
        super(Loss, self).__init__()
        self.random_distractors = random_distractors

    def forward(
        self,
        _sender_input,
        _message,
        _receiver_input,
        receiver_output,
        _labels,
        aux_input,
    ):
        labels = aux_input["game_labels"].view(-1)
        mask = aux_input["mask"].float().view(-1)

        if not self.training and self.random_distractors:
            bsz, max_objs = aux_input["mask"].shape
            acc_labels = torch.zeros(bsz, device=receiver_output.device)
            acc = (
                (receiver_output[0::max_objs].argmax(dim=-1) == acc_labels)
                .detach()
                .float()
            )
        else:
            acc = (receiver_output.argmax(dim=-1) == labels).detach().float()
            acc *= mask  # zeroing masked elements
            acc = (acc.sum() / mask.sum()).unsqueeze(0)  # avoid dimensionless tensors

        loss = F.cross_entropy(receiver_output, labels, reduction="none")
        loss *= mask  # multiply by 0 masked elements

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
        attn_type = no_attention
    else:
        raise NotImplementedError
    return attn_type


def build_sender(opts):
    return Sender(
        input_dim=opts.img_feats_dim,
        output_dim=opts.vocab_size,
        attn_fn=build_attention(opts),
        temperature=opts.gs_temperature,
        random_ctx_position=opts.random_ctx_position,
        sender_separate_messages=opts.sender_separate_messages,
    )


def build_receiver(opts):
    receiver = Receiver(
        input_dim=opts.img_feats_dim,
        output_dim=opts.output_dim,
        temperature=opts.loss_temperature,
    )
    args = [receiver, opts.vocab_size, opts.output_dim, opts.recv_separate_embeddings]
    if opts.attn_type == "none":
        return SingleSymbolReceiverWrapper(*args)
    return DoubleSymbolReceiverWrapper(*args)


def build_game(opts):
    visual_encoder = get_cnn(opts)
    sender = build_sender(opts)
    receiver = build_receiver(opts)

    game = SymbolGameGS(
        sender=sender,
        receiver=receiver,
        loss=Loss(opts.random_distractors),
        train_logging_strategy=LoggingStrategy.minimal(),
        test_logging_strategy=LoggingStrategy.minimal(),
    )
    if opts.distributed_context.is_distributed:
        game = nn.SyncBatchNorm.convert_sync_batchnorm(game)
    return VisionWrapper(game, visual_encoder)
