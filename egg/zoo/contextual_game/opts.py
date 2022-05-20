# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import egg.core as core


def get_data_opts(parser):
    group = parser.add_argument_group("data options")
    group.add_argument("--image_size", type=int, default=224, help="Image size")
    group.add_argument(
        "--image_dir", default="/private/home/rdessi/imagecode/data/images"
    )
    group.add_argument(
        "--metadata_dir",
        default="/private/home/rdessi/exp_EGG/egg/zoo/contextual_game/dataset",
    )
    group.add_argument("--num_workers", type=int, default=8)


def get_clip_opts(parser):
    group = parser.add_argument_group("clip opts")
    group.add_argument("--max_clip_vocab", type=int, default=None)


def get_game_opts(parser):
    group = parser.add_argument_group("game opts")
    # multi symbol RNN options
    group.add_argument(
        "--cell",
        choices=["rnn", "gru"],
        default="rnn",
        help="Type of the cell used for Sender {rnn, gru} (default: rnn)",
    )
    group.add_argument("--sender_embed_dim", type=int, default=256)
    group.add_argument("--num_layers", default=1, type=int)


def get_common_opts(params):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Run the game with pdb enabled",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        default=False,
        help="Skip training and evaluate from checkopoint",
    )
    parser.add_argument(
        "--gs_temperature",
        type=float,
        default=1.0,
        help="gs temperature used in the relaxation layer",
    )
    parser.add_argument(
        "--straight_through",
        default=False,
        action="store_true",
        help="use straight through gumbel softmax estimator",
    )
    parser.add_argument("--finetune", default=False, action="store_true")
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="Enable wandb logging",
    )
    parser.add_argument("--wandb_tag", default="default")
    parser.add_argument("--wandb_project", default="playground")

    get_data_opts(parser)
    get_game_opts(parser)
    get_clip_opts(parser)

    opts = core.init(arg_parser=parser, params=params)
    return opts
