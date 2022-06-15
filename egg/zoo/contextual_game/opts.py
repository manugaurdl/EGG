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


def get_clipclap_opts(parser):
    group = parser.add_argument_group("clipclap opts")
    group.add_argument("--clipclap_model_path", default=None)
    group.add_argument("--mapping_type", choices=["mlp", "transformer"], default="mlp")
    group.add_argument("--clip_prefix_tokens", type=int, default=10)
    group.add_argument("--constant_prefix_tokens", type=int, default=10)
    group.add_argument("--num_transformer_layers", type=int, default=8)
    group.add_argument(
        "--clip_model", choices=["ViT-B/32", "RN50x4"], default="ViT-B/32"
    )
    group.add_argument("--use_beam_search", action="store_true", default=False)
    group.add_argument("--num_beams", type=int, default=5)
    group.add_argument("--prefix_only", action="store_true", default=False)


def get_common_opts(params):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Run the game with pdb enabled",
    )
    parser.add_argument("--warmup_steps", type=int, default=5000)

    get_data_opts(parser)
    get_clipclap_opts(parser)

    opts = core.init(arg_parser=parser, params=params)
    return opts
