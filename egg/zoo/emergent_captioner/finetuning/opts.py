# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import egg.core as core


def get_data_opts(parser):
    group = parser.add_argument_group("data options")
    group.add_argument("--dataset_dir", default=None)
    group.add_argument(
        "--dataset",
        choices=["coco", "flickr", "conceptual"],
        default="coco",
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
    group.add_argument("--freeze_clipcap_mapper", action="store_true", default=False)
    group.add_argument("--num_hard_negatives", type=int, default=0)
    group.add_argument("--in_batch_negatives", action="store_true", default=False)
    group.add_argument("--test_w_negatives", action="store_true", default=False)

    group.add_argument(
        "--beam_size",
        type=int,
        default=5,
        help="Number of beams when using beam serach decoding",
    )
    group.add_argument("--num_return_sequences", type=int, default=1)
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
    group.add_argument(
        "--sender_entropy_coeff",
        type=float,
        default=0.0,
        help="The entropy regularization coefficient for Sender used in reinforce",
    )
    group.add_argument("--kl_div_coeff", type=float, default=0.0)


def get_optimizer_opts(parser):
    group = parser.add_argument_group("optimizer options")

    group.add_argument(
        "--opt",
        default="adam",
        choices=["adam", "adamw"],
    )
    group.add_argument("--opt_scheduler", action="store_true", default=False)
    group.add_argument("--warmup_steps", type=int, default=1000)


def get_common_opts(params):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--eval_out_of_the_box",
        action="store_true",
        default=False,
        help="Performa an evaluation loop before training the model",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        default=False,
        help="Run only the evaluation loop on the test",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Run the game with pdb enabled",
    )

    get_data_opts(parser)
    get_clipcap_opts(parser)
    get_optimizer_opts(parser)
    get_game_opts(parser)

    opts = core.init(arg_parser=parser, params=params)
    return opts
