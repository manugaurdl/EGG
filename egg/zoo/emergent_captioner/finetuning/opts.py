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
        default=["flickr"],
    )
    group.add_argument(
        "--sender_image_size", type=int, default=224, help="Sender Image size"
    )
    group.add_argument(
        "--recv_image_size", type=int, default=None, help="Recv image size"
    )
    group.add_argument("--num_workers", type=int, default=8)


def get_captioner_opts(parser):
    group = parser.add_argument_group("captioner options")

    group.add_argument(
        "--captioner_model",
        choices="clipcap blip camel".split(),
        default="clipcap",
        help="Type of captioner model (default: clipcap)",
    )

    # Blip
    group.add_argument(
        "--blip_model",
        choices=["base_coco", "large_coco"],
        default="base_coco",
    )
    group.add_argument(
        "--freeze_blip_visual_encoder", action="store_true", default=False
    )

    # Clipcap
    group.add_argument(
        "--clipcap_model_path",
        default="/private/home/rdessi/EGG/egg/zoo/emergent_captioner/clipclap_models/conceptual_weights.pt",
    )
    group.add_argument(
        "--sender_clip_model",
        choices=["ViT-B/16", "ViT-B/32", "RN50x16"],
        default="ViT-B/32",
    )

    # Camel
    group.add_argument(
        "--camel_model_path",
        type=str,
        default="/private/home/rdessi/EGG/egg/zoo/emergent_captioner/finetuning/camel_models/checkpoints/camel_nomesh.pth",
    )

    group.add_argument(
        "--network", type=str, choices=("online", "target"), default="target"
    )
    group.add_argument("--disable_mesh", action="store_true")

    group.add_argument("--N_dec", type=int, default=3)
    group.add_argument("--N_enc", type=int, default=3)
    group.add_argument("--d_model", type=int, default=512)
    group.add_argument("--d_ff", type=int, default=2048)
    group.add_argument("--m", type=int, default=40)
    group.add_argument("--head", type=int, default=8)
    group.add_argument("--with_pe", action="store_true")

    # Other
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
    get_captioner_opts(parser)
    get_game_opts(parser)

    opts = core.init(arg_parser=parser, params=params)
    return opts
