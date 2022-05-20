# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from collections import Counter
from pathlib import Path

import clip
import torch
import torch.nn.functional as F

from egg.core.gs_wrappers import RelaxedEmbedding
from egg.zoo.contextual_game.archs import (
    ClipReceiver,
    InformedRnnSenderFixedLengthGS,
    VisionGame,
)


def loss(
    _sender_input,
    _message,
    _receiver_input,
    receiver_output,
    _labels,
    _aux_input,
):

    labels = torch.arange(receiver_output.shape[0]).to(receiver_output.device)

    acc = (receiver_output.argmax(dim=1) == labels).detach().float()
    loss = F.cross_entropy(receiver_output, labels, reduction="none")

    captioned_acc = (receiver_output[_labels].argmax(dim=1) == _labels).detach().float()
    return loss, {"acc": acc, "captioned_acc": captioned_acc}


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()


def extract_most_frequent_embeddings(
    pretrained_embeddings: torch.Tensor,
    max_vocab: int = None,
    data_path: str = "/private/home/rdessi/imagecode/data/",
):
    data_path = Path(data_path)

    assert max_vocab is None or max_vocab > 0

    # not including the test set since it is unlabeled and not used
    with open(data_path / "train_data.json") as fd:
        train = json.load(fd)
    with open(data_path / "valid_data.json") as fd:
        valid = json.load(fd)

    train_and_valid = {**train, **valid}

    token_list = []
    for _, captions in train_and_valid.items():
        for caption in captions.values():
            token_list.extend(clip.tokenize(caption, truncate=True)[0].tolist())

    token_counter = Counter(token_list)

    max_vocab = max_vocab if max_vocab else len(token_counter)
    # adding eos, sos and pad at the end
    most_freq_tokens = [
        x[0]
        for x in token_counter.most_common(max_vocab)
        if x[0] not in [49406, 49407, 0]
    ] + [0, 49406, 49407]

    embeddings = pretrained_embeddings.weight[most_freq_tokens]
    return embeddings, embeddings.shape[1]


def build_game(opts):
    clip_model = clip.load("ViT-B/16")[0]

    clip_model.train()
    convert_models_to_fp32(clip_model)

    if not opts.finetune:
        clip_model.eval()
        for param in clip_model.parameters():
            param.requires_grad = False

    embeddings, clip_embed_dim = extract_most_frequent_embeddings(
        clip_model.token_embedding,
        opts.max_clip_vocab,
    )

    embeddings = RelaxedEmbedding.from_pretrained(embeddings, freeze=not opts.finetune)
    clip_model.token_embedding = embeddings
    sender = InformedRnnSenderFixedLengthGS(
        input_dim=clip_model.visual.output_dim,
        num_encoder_layers=opts.num_layers,
        vocab_size=opts.max_clip_vocab,
        embed_dim=opts.sender_embed_dim,
        hidden_size=clip_embed_dim,
        max_len=opts.max_len,
        embeddings=embeddings,
        cell=opts.cell,
        temperature=opts.gs_temperature,
        straight_through=opts.straight_through,
    )

    receiver = ClipReceiver(
        clip_model,
        pad_idx=opts.max_clip_vocab - 3,
        sos_idx=opts.max_clip_vocab - 2,
        eos_idx=opts.max_clip_vocab - 1,
    )

    game = VisionGame(sender, receiver, loss)
    if opts.distributed_context.is_distributed:
        game = torch.nn.SyncBatchNorm.convert_sync_batchnorm(game)

    return game
