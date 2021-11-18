# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging

import egg.core as core


def get_data_opts(parser):
    group = parser.add_argument_group("data")
    group.add_argument(
        "--dataset_dir",
        type=str,
        default="/datasets01/VisualGenome1.2/061517",
        help="Dataset location",
    )
    group.add_argument("--image_size", type=int, default=64, help="Image size")
    group.add_argument("--use_augmentations", action="store_true", default=False)


def get_gs_opts(parser):
    group = parser.add_argument_group("gumbel softmax")
    group.add_argument(
        "--gs_temperature",
        type=float,
        default=1.0,
        help="gs temperature used in the relaxation layer",
    )


def get_vision_module_opts(parser):
    group = parser.add_argument_group("vision module")
    group.add_argument(
        "--vision_model_name",
        type=str,
        default="resnet101",
        choices=["resnet50", "resnet101", "resnet152"],
        help="Model name for the encoder",
    )
    group.add_argument(
        "--pretrain_vision",
        default=False,
        action="store_true",
        help="If set, pretrained vision modules will be used",
    )
    group.add_argument(
        "--shared_vision",
        default=False,
        action="store_true",
        help="If set, the vision module will be shared by sender and reciver",
    )


def get_game_arch_opts(parser):
    group = parser.add_argument_group("game architecture")
    group.add_argument(
        "--recv_temperature",
        type=float,
        default=0.1,
        help="Temperature for similarity computation in the loss fn. Ignored when similarity is 'dot'",
    )
    group.add_argument(
        "--recv_hidden_dim",
        type=int,
        default=2048,
        help="Hidden dim of the non-linear projection of the distractors",
    )
    group.add_argument(
        "--recv_output_dim",
        type=int,
        default=2048,
        help="Output dim of the non-linear projection of the distractors, used to compare with msg embedding",
    )


def get_common_opts(params):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Run the game with pdb enabled",
    )

    get_data_opts(parser)
    get_gs_opts(parser)
    get_vision_module_opts(parser)
    get_game_arch_opts(parser)

    opts = core.init(arg_parser=parser, params=params)
    if opts.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level)

    return opts
