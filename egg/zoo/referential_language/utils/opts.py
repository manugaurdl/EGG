# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import egg.core as core


def get_data_opts(parser):
    group = parser.add_argument_group("data options")
    group.add_argument(
        "--dataset_name",
        type=str,
        default="flickr",
        choices=["flickr", "vg"],
        help="Dataset to use for game playing",
    )
    group.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Path to folder with images",
    )
    group.add_argument(
        "--metadata_dir",
        type=str,
        default=None,
        help="Path to folder with images metada",
    )
    group.add_argument(
        "--max_objects",
        type=int,
        default=10,
        help="Max number of bboxes to extract from an image",
    )
    group.add_argument("--image_size", type=int, default=64, help="Image size")
    group.add_argument("--random_distractors", default=False, action="store_true")
    group.add_argument("--use_augmentation", default=False, action="store_true")


def get_vision_module_opts(parser):
    group = parser.add_argument_group("vision module options")
    group.add_argument(
        "--vision_model",
        type=str,
        default="resnet34",
        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
        help="Model name for the encoder",
    )
    group.add_argument(
        "--pretrain_vision",
        default=False,
        action="store_true",
        help="If set, pretrained vision models will be used",
    )


def get_attention_opts(parser):
    group = parser.add_argument_group("sender attention options")
    group.add_argument(
        "--attn_fn",
        default="none",
        choices=["none", "dot", "self", "top", "target", "random", "random_context"],
    )
    group.add_argument(
        "--attn_topk",
        default=1,
        type=int,
        help="Use top k most similar distractors to compute the attention vector",
    )
    group.add_argument(
        "--num_heads",
        default=1,
        type=int,
        help="Number of attention heads used in self attention",
    )
    group.add_argument(
        "--random_k",
        default=False,
        action="store_true",
        help="Use k random distractors to compute the attention vector",
    )


def get_game_arch_opts(parser):
    group = parser.add_argument_group("game architecture options")
    group.add_argument(
        "--loss_temperature",
        type=float,
        default=1.0,
        help="Temperature for similarity computation in the loss fn. Ignored when similarity is 'dot'",
    )
    group.add_argument(
        "--output_dim",
        type=int,
        default=1024,
        help="Output dim of the non-linear projection of the distractors, used to compare with msg embedding",
    )
    group.add_argument(
        "--sender_embed_dim",
        type=int,
        default=256,
        help="Output dim of rnn embeddings when using rnn to produce and read messages",
    )
    group.add_argument(
        "--sender_cell",
        default="rnn",
        choices=["rnn", "lstm", "gru"],
    )
    group.add_argument(
        "--recv_embed_dim",
        type=int,
        default=256,
        help="Output dim of rnn embeddings when using rnn to produce and read messages",
    )
    group.add_argument(
        "--recv_cell_hidden_size",
        type=int,
        default=256,
    )
    group.add_argument(
        "--recv_cell",
        default="rnn",
        choices=["rnn", "lstm", "gru"],
    )


def get_gs_opts(parser):
    group = parser.add_argument_group("gumbel softmax training options")
    group.add_argument(
        "--gs_temperature",
        type=float,
        default=1.0,
        help="gs temperature used in the relaxation layer",
    )


def get_common_opts(params):
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False)

    get_data_opts(parser)
    get_gs_opts(parser)
    get_vision_module_opts(parser)
    get_game_arch_opts(parser)
    get_attention_opts(parser)

    opts = core.init(arg_parser=parser, params=params)
    return opts
