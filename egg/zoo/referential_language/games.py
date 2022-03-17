# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from egg.core.gs_wrappers import SymbolGameGS
from egg.zoo.referential_language.archs import (
    Attention_topk,
    SingleSymbolReceiverWrapper,
    DoubleSymbolReceiverWrapper,
    MessageGeneratorRnn,
    MessageGeneratorMLP,
    NoAttention,
    Receiver,
    RnnReceiver,
    ScaledDotProductAttention,
    Sender,
    SelfAttention,
    TargetAttention,
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
    bsz, max_objs = aux_input["mask"].shape
    labels = torch.arange(max_objs, device=receiver_output.device)
    labels = labels.repeat(bsz, 1).view(-1)
    mask = aux_input["mask"].float().view(-1)

    acc = (receiver_output.argmax(dim=-1) == labels).detach().float()
    aux_input["guesses"] = acc.view(bsz, max_objs, -1)

    # multiply by 0-masked elements
    loss = F.cross_entropy(receiver_output, labels, reduction="none") * mask
    avg_acc = ((acc * mask).sum() / mask.sum()).unsqueeze(0)
    return loss, {"acc": avg_acc, "baseline": 1 / aux_input["mask"].int().sum(-1)}


def build_attention(opts):
    if opts.attn_type == "top":
        attn_fn = Attention_topk(
            img_feats_dim=opts.img_feats_dim, k=opts.attn_topk, random=opts.random_k
        )
    elif opts.attn_type == "dot":
        attn_fn = ScaledDotProductAttention(img_feats_dim=opts.img_feats_dim)
    elif opts.attn_type == "self":
        attn_fn = SelfAttention(embed_dim=opts.img_feats_dim, num_heads=opts.num_heads)
    elif opts.attn_type == "none":
        if not opts.single_symbol:
            raise RuntimeError(
                "Cannot have double symbol w/o attention, use single symbol or TargetAttention instead"
            )
        attn_fn = NoAttention()
    elif opts.attn_type == "target":
        attn_fn = TargetAttention()
    else:
        raise NotImplementedError
    return attn_fn


def build_message_generator(opts):
    if opts.message_model == "mlp":
        input_dim = opts.img_feats_dim * 2 if opts.cat_ctx else opts.img_feats_dim
        return MessageGeneratorMLP(
            input_dim=input_dim,
            output_dim=opts.vocab_size,
            single_symbol=opts.single_symbol,
            temperature=opts.gs_temperature,
            cat_ctx=opts.cat_ctx,
            shuffle_cat=opts.shuffle_cat,
            separate_mlps=opts.sender_separate_mlps,
        )
    elif opts.message_model == "rnn":
        hidden_size = opts.img_feats_dim * 2 if opts.cat_ctx else opts.img_feats_dim
        return MessageGeneratorRnn(
            opts.vocab_size,
            opts.sender_embed_dim,
            hidden_size,
            opts.cat_ctx,
            opts.shuffle_cat,
            opts.gs_temperature,
            opts.sender_cell,
        )
    else:
        raise RuntimeError(f"Cannot recognize model {opts.message_model}")


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
    if opts.message_model == "mlp":
        args = [
            receiver,
            opts.vocab_size,
            opts.output_dim,
            opts.recv_separate_embeddings,
        ]
        if opts.single_symbol:
            return SingleSymbolReceiverWrapper(*args)
        return DoubleSymbolReceiverWrapper(*args)
    elif opts.message_model == "rnn":
        return RnnReceiver(
            receiver,
            opts.vocab_size,
            opts.recv_embed_dim,
            opts.recv_cell_hidden_size,
            opts.output_dim,
            opts.recv_cell,
        )
    else:
        raise RuntimeError(f"Cannot recognize model {opts.message_model}")


def build_game(opts):
    visual_encoder = get_cnn(opts)
    sender = build_sender(opts)
    receiver = build_receiver(opts)

    game = SymbolGameGS(sender=sender, receiver=receiver, loss=loss)
    if opts.distributed_context.is_distributed:
        game = nn.SyncBatchNorm.convert_sync_batchnorm(game)

    return VisionWrapper(game, visual_encoder)
