# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy

import clip
import torch
import torch.nn.functional as F

from egg.zoo.contextual_game.archs import (
    ClipEmbeddingLoader,
    ClipReceiver,
    RnnSenderFixedLengthGS,
    Sender,
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

    loss = F.cross_entropy(captioned_img, labels, reduction="none")

    return loss, {"acc": captioned_img_acc, "all_accs": all_imgs_acc}


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


def get_clip_embeddings(
    clip_model,
    freeze_sender_embeddings,
    freeze_recv_embeddings,
    share_embeddings,
    max_clip_vocab: int = 3000,
):
    embeddings = clip_model.token_embedding
    embedding_loader = ClipEmbeddingLoader(embeddings, max_vocab=max_clip_vocab)

    pretrained_embeddings = embedding_loader.embeddings
    vocab_size = embedding_loader.vocab_size

    if freeze_sender_embeddings and freeze_recv_embeddings:
        pretrained_embeddings.weight.requires_grad = False
        sender_embeddings = recv_embeddings = pretrained_embeddings
    elif not (freeze_sender_embeddings and freeze_recv_embeddings):
        pretrained_embeddings.weight.requires_grad = True
        sender_embeddings = pretrained_embeddings

        recv_embeddings = (
            pretrained_embeddings
            if share_embeddings
            else copy.deepcopy(pretrained_embeddings)
        )
    else:
        sender_embeddings = pretrained_embeddings
        recv_embeddings = copy.deepcopy(pretrained_embeddings)

        sender_embeddings.weight.requires_grad = not freeze_sender_embeddings
        recv_embeddings.weight.requires_grad = not freeze_recv_embeddings

    return sender_embeddings, recv_embeddings, vocab_size


def get_visual_encoders(
    visual_encoder,
    freeze_sender_encoder,
    freeze_recv_encoder,
    share_encoders,
):
    if freeze_sender_encoder and freeze_recv_encoder:
        visual_encoder.eval()
        for param in visual_encoder.parameters():
            param.requires_grad = False
        sender_visual_encoder = recv_visual_encoder = visual_encoder
    elif not (freeze_sender_encoder and freeze_recv_encoder):
        for param in visual_encoder.parameters():
            param.requires_grad = False

        sender_visual_encoder = visual_encoder
        recv_visual_encoder = (
            visual_encoder if share_encoders else copy.deepcopy(visual_encoder)
        )
    else:
        sender_visual_encoder = visual_encoder
        recv_visual_encoder = copy.deepcopy(visual_encoder)

        for param in sender_visual_encoder.parameters():
            param.requires_grad = not freeze_sender_encoder
        for param in recv_visual_encoder.parameters():
            param.requires_grad = not freeze_recv_encoder
    return sender_visual_encoder, recv_visual_encoder


def build_game(opts):
    clip_model = initialize_clip(opts.vision_model)

    vision_model = clip_model.visual

    sender_encoder, recv_encoder = get_visual_encoders(
        vision_model,
        opts.freeze_sender_encoder,
        opts.freeze_recv_encoder,
        opts.share_visual_encoders,
    )

    sender_emb, recv_emb, vocab_size = get_clip_embeddings(
        clip_model,
        opts.freeze_sender_embeddings,
        opts.freeze_recv_embeddings,
        opts.share_embeddings,
        opts.max_clip_vocab,
    )

    sender = RnnSenderFixedLengthGS(
        agent=Sender(
            input_dim=clip_model.visual.output_dim,
            output_dim=opts.sender_rnn_hidden_size,
        ),
        temperature=opts.gs_temperature,
        straight_through=opts.straight_through,
        vocab_size=vocab_size,
        embed_dim=opts.sender_rnn_embed_dim,
        hidden_size=opts.sender_rnn_hidden_size,
        max_len=opts.max_len,
        embeddings=sender_emb,
        cell=opts.sender_cell,
    )

    receiver = ClipReceiver(clip_model, recv_emb, finetune_weights=opts.finetune_clip)

    game = VisionGame(sender_encoder, recv_encoder, sender, receiver, loss)
    if opts.distributed_context.is_distributed:
        game = torch.nn.SyncBatchNorm.convert_sync_batchnorm(game)

    return game
