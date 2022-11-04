# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import egg.core as core


def get_data_opts(parser):
    group = parser.add_argument_group("data options")
    group.add_argument("--dataset_dir", default=None)
    group.add_argument(
        "--train_dataset",
        choices=["coco", "conceptual"],
        default="coco",
    )
    group.add_argument(
        "--eval_datasets",
        nargs="+",
        default=["coco"],
    )
    group.add_argument("--image_size", type=int, default=224, help="Image size")
    group.add_argument("--num_workers", type=int, default=8)


def get_clipcap_opts(parser):
    group = parser.add_argument_group("clipcap options")

    group.add_argument(
        "--clipcap_model_path",
        default="/private/home/rdessi/EGG/egg/zoo/emergent_captioner/clipclap_models/conceptual_weights.pt",
    )
    group.add_argument(
        "--sender_clip_model",
        choices=["ViT-B/16", "ViT-B/32"],
        default="ViT-B/32",
    )
    group.add_argument("--num_hard_negatives", type=int, default=0)

    group.add_argument(
        "--beam_size",
        type=int,
        default=1,
        help="Number of beams when using beam serach decoding",
    )
    group.add_argument("--do_sample", action="store_true", default=False)


def get_game_opts(parser):
    group = parser.add_argument_group("game options")

    group.add_argument(
        "--loss_type",
        choices="accuracy similarity discriminative".split(),
        default="discriminative",
    )
    group.add_argument(
        "--recv_clip_model",
        choices=["ViT-B/16", "ViT-B/32"],
        default="ViT-B/32",
    )
    group.add_argument(
        "--baseline",
        choices=["no", "mean"],
        default="no",
    )
    group.add_argument("--kl_div_coeff", type=float, default=0.0)


def get_common_opts(params):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Run the game with pdb enabled",
    )

    get_data_opts(parser)
    get_clipcap_opts(parser)
    get_game_opts(parser)

    opts = core.init(arg_parser=parser, params=params)
    return opts
