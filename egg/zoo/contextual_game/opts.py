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


def get_bart_opts(parser):
    group = parser.add_argument_group("bart opts")

    group.add_argument(
        "--bart_model",
        choices=["facebook/bart-base", "eugenesiow/bart-paraphrase"],
        default="eugenesiow/bart-paraphrase",
    )
    group.add_argument("--num_beams", type=int, default=4)


def get_common_opts(params):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Run the game with pdb enabled",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="Enable wandb logging",
    )
    parser.add_argument("--wandb_tag", default="default")
    parser.add_argument("--wandb_project", default="playground")

    get_data_opts(parser)
    get_bart_opts(parser)

    opts = core.init(arg_parser=parser, params=params)
    return opts
