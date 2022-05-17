# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F

from egg.core.gs_wrappers import RelaxedEmbedding
from egg.zoo.contextual_game.archs import (
    ClipEmbeddingLoader,
    ClipReceiver,
    InformedRnnSenderFixedLengthGS,
    RnnSenderFixedLengthGS,
    Sender,
    SymbolSender,
    VisionGame,
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

    loss = F.cross_entropy(receiver_output, all_img_labels, reduction="none")

    return loss, {"acc": all_imgs_acc, "captioned_img_acc": captioned_img_acc}


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()


def initialize_clip(name: str = "ViT-B/16"):
    modules = {
        "clip_vit_b/32": clip.load("ViT-B/32")[0],
        "clip_vit_b/16": clip.load("ViT-B/16")[0],
        "clip_vit_l/14": clip.load("ViT-L/14")[0],
        "clip_resnet50": clip.load("RN50")[0],
        "clip_resnet101": clip.load("RN101")[0],
    }

    if name not in modules:
        raise KeyError(f"{name} is not currently supported.")

    model = modules[name].train()

    convert_models_to_fp32(model)

    return model


def get_clip_embeddings(embeddings, freeze_recv, max_clip_vocab: int = 3000):
    sender_embeddings = ClipEmbeddingLoader(
        embeddings, max_vocab=max_clip_vocab
    ).embeddings
    recv_embeddings = ClipEmbeddingLoader(
        embeddings, freeze_recv, max_vocab=max_clip_vocab, include_special_symbols=True
    ).embeddings

    return sender_embeddings, recv_embeddings, sender_embeddings.weight.shape[1]


def get_visual_encoder(visual_encoder, finetune):
    if finetune:
        visual_encoder.train()
        return visual_encoder

    visual_encoder.eval()
    for param in visual_encoder.parameters():
        param.requires_grad = False
    return visual_encoder


def build_game(opts):
    clip_model = initialize_clip(opts.vision_model)

    visual_encoder = get_visual_encoder(clip_model.visual, opts.finetune)

    sender_emb, recv_emb, clip_embed_dim = get_clip_embeddings(
        clip_model.token_embedding,
        not opts.finetune,
        opts.max_clip_vocab,
    )

    if opts.max_len == 1:
        if opts.informed_sender:
            encoder = Sender(visual_encoder.output_dim, clip_embed_dim, opts.num_layers)
            sender_emb = RelaxedEmbedding.from_pretrained(
                sender_emb.weight.t(), freeze=False
            )
        else:
            encoder = Sender(
                visual_encoder.output_dim, opts.max_clip_vocab, opts.num_layers
            )
            sender_emb = nn.Identity()
        sender = SymbolSender(
            encoder,
            sender_emb,
            temperature=opts.gs_temperature,
            straight_through=opts.straight_through,
        )
    else:
        encoder = Sender(visual_encoder.output_dim, clip_embed_dim, opts.num_layers)
        if opts.informed_sender:
            sender_wrapper = InformedRnnSenderFixedLengthGS
            sender_emb = RelaxedEmbedding.from_pretrained(
                sender_emb.weight.t(), freeze=False
            )
        else:
            sender_wrapper = RnnSenderFixedLengthGS

        sender = sender_wrapper(
            encoder,
            temperature=opts.gs_temperature,
            straight_through=opts.straight_through,
            vocab_size=opts.max_clip_vocab,
            embed_dim=clip_embed_dim,
            hidden_size=clip_embed_dim,
            max_len=opts.max_len,
            embeddings=sender_emb,
            cell=opts.cell,
        )

    vocab_size = opts.max_clip_vocab if opts.max_clip_vocab else 49405
    receiver = ClipReceiver(
        clip_model,
        recv_emb,
        finetune_weights=opts.finetune,
        pad_idx=vocab_size,
        sos_idx=vocab_size + 1,
        eos_idx=vocab_size + 2,
    )

    game = VisionGame(visual_encoder, sender, receiver, loss)
    if opts.distributed_context.is_distributed:
        game = torch.nn.SyncBatchNorm.convert_sync_batchnorm(game)

    return game
