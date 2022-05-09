# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from egg.zoo.contextual_game.archs import Receiver, Sender
from egg.zoo.contextual_game.wrappers import (
    ClipEmbeddingLoader,
    ClipReceiver,
    GumbelSoftmaxWrapper,
    RnnReceiverFixedLengthGS,
    RnnSenderFixedLengthGS,
    SymbolReceiverWrapper,
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


def initialize_clip(name: str = "ViT-B/16", pretrained: bool = False):
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

    n_features = model.visual.output_dim
    convert_models_to_fp32(model)

    if pretrained:
        for name, param in model.named_parameters():
            if "visual" in name:
                param.requires_grad = False
        model = model.eval()

    return model, n_features


def initialize_resnet(name: str = "resnet50", pretrained: bool = False):
    modules = {
        "resnet50": torchvision.models.resnet50(pretrained=pretrained),
        "resnet101": torchvision.models.resnet101(pretrained=pretrained),
        "resnet152": torchvision.models.resnet152(pretrained=pretrained),
        "vgg11": torchvision.models.vgg11(pretrained=pretrained),
    }

    if name not in modules:
        raise KeyError(f"{name} is not currently supported.")

    model = modules[name]

    if name in ["resnet50", "resnet101", "resnet152"]:
        n_features = model.fc.in_features
        model.fc = nn.Identity()

    else:  # vgg11
        n_features = model.classifier[6].in_features
        model.classifier[6] = nn.Identity()

    if pretrained:
        for param in model.parameters():
            param.requires_grad = False
        model = model.eval()

    return model, n_features


def build_sender(visual_feats_size, vocab_size, pretrained_embeddings, opts):
    sender = Sender(
        input_dim=visual_feats_size,
        output_dim=vocab_size if opts.max_len == 1 else opts.sender_rnn_hidden_size,
    )

    len2sender = {1: GumbelSoftmaxWrapper}

    sender = len2sender.get(opts.max_len, RnnSenderFixedLengthGS)(
        sender,
        temperature=opts.gs_temperature,
        straight_through=opts.straight_through,
        vocab_size=vocab_size,
        embed_dim=opts.sender_rnn_embed_dim,
        hidden_size=opts.sender_rnn_hidden_size,
        max_len=opts.max_len,
        embeddings=pretrained_embeddings if opts.sender_clip_embeddings else None,
        freeze_embeddings=opts.freeze_clip_embeddings,
        cell=opts.sender_cell,
    )

    print(f"| Using {opts.max_len} symbols, sender of class {type(sender).__name__}")

    return sender


def build_receiver(
    visual_feats_size, vocab_size, clip_model, pretrained_embeddings, opts
):
    receiver = Receiver(
        input_dim=visual_feats_size,
        hidden_dim=opts.recv_hidden_dim,
        output_dim=opts.recv_output_dim,
        use_mlp=opts.use_mlp_recv,
        temperature=opts.loss_temperature,
    )
    if opts.clip_receiver:
        receiver = ClipReceiver(
            receiver,
            model=clip_model,
            embeddings=pretrained_embeddings,
            add_clip_tokens=opts.add_clip_tokens,
            finetune_weights=opts.finetune_clip,
            freeze_embeddings=opts.freeze_clip_embeddings,
            max_clip_vocab=opts.max_clip_vocab,
        )
    else:
        if opts.max_len == 1:
            receiver = SymbolReceiverWrapper(
                receiver,
                vocab_size=vocab_size,
                embeddings=pretrained_embeddings,
                agent_input_size=opts.recv_output_dim,
            )
        else:
            receiver = RnnReceiverFixedLengthGS(
                receiver,
                vocab_size=vocab_size,
                embeddings=pretrained_embeddings if opts.recv_clip_embeddings else None,
                embed_dim=opts.recv_rnn_embed_dim,
                hidden_size=opts.recv_rnn_hidden_size,
                cell=opts.sender_cell,
                freeze_embeddings=opts.freeze_clip_embeddings,
            )
    print(f"| Using {opts.max_len} symbols, recv of class {type(receiver).__name__}")
    return receiver


def build_game(opts):
    clip_model = None
    if "clip" in opts.vision_model:
        clip_model, visual_feats_size = initialize_clip(
            opts.vision_model, opts.pretrain_vision
        )
        vision_model = clip_model.visual
    else:
        vision_model, visual_feats_size = initialize_resnet(
            opts.vision_model, opts.pretrain_vision
        )

    if opts.sender_clip_embeddings or opts.recv_clip_embeddings or opts.clip_receiver:
        embedding_loader = ClipEmbeddingLoader(
            clip_model, opts.freeze_clip_embeddings, max_vocab=opts.max_clip_vocab
        )
        print("| Done loading clip embeddings")

        pretrained_embeddings = embedding_loader.embeddings
        vocab_size = embedding_loader.vocab_size
    else:
        pretrained_embeddings = None
        vocab_size = opts.vocab_size

    assert opts.max_len > 0

    sender = build_sender(visual_feats_size, vocab_size, pretrained_embeddings, opts)
    receiver = build_receiver(
        visual_feats_size, vocab_size, clip_model, pretrained_embeddings, opts
    )
    game = VisionGame(vision_model, sender, receiver, loss)
    if opts.distributed_context.is_distributed:
        game = torch.nn.SyncBatchNorm.convert_sync_batchnorm(game)

    return game
