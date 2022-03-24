# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from egg.core.gs_wrappers import RelaxedEmbedding, SymbolGameGS
from egg.zoo.referential_language import modules
from egg.zoo.referential_language.modules.utils import get_cnn


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
    name2attn = {
        "top": modules.AttentionTopK,
        "dot": modules.ScaledDotProductAttention,
        "self": modules.MHAttention,
        "target": modules.TargetAttention,
        "none": modules.NoAttention,
    }
    attn_fn = name2attn.get(opts.attn_fn, None)

    if attn_fn is None:
        raise KeyError(f"Cannot recognize attn {opts.attn_fn}")

    return attn_fn(embed_dim=opts.img_feats_dim, **vars(opts))


def build_message_generator(opts):
    input_dim = opts.img_feats_dim
    if opts.attn_fn != "none":
        input_dim *= 2
    return modules.CatMLP(input_dim, opts.vocab_size, opts.gs_temperature)


def build_sender(opts):
    return modules.Sender(
        attn_fn=build_attention(opts), msg_generator=build_message_generator(opts)
    )


def build_receiver(opts):
    linear_reader = RelaxedEmbedding(opts.vocab_size, opts.output_dim)
    return modules.Receiver(
        input_dim=opts.img_feats_dim,
        output_dim=opts.output_dim,
        temperature=opts.loss_temperature,
        msg_reader=linear_reader,
    )


def build_game(opts):
    visual_encoder = get_cnn(opts)
    sender = build_sender(opts)
    receiver = build_receiver(opts)

    game = SymbolGameGS(sender=sender, receiver=receiver, loss=loss)
    if opts.distributed_context.is_distributed:
        game = nn.SyncBatchNorm.convert_sync_batchnorm(game)

    return modules.VisionWrapper(game, visual_encoder)
