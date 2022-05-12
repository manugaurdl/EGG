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


def get_vision_model_opts(parser):
    group = parser.add_argument_group("vision model options")
    group.add_argument(
        "--vision_model",
        type=str,
        default="clip_vit_b/16",
        choices=[
            "clip_vit_b/32",
            "clip_vit_b/16",
            "clip_vit_l/14",
            "clip_resnet50",
            "clip_resnet101",
        ],
        help="Model name for the visual encoder",
    )
    group.add_argument("--freeze_sender_encoder", default=False, action="store_true")
    group.add_argument("--freeze_recv_encoder", default=False, action="store_true")
    group.add_argument("--share_visual_encoders", default=False, action="store_true")


def get_clip_opts(parser):
    group = parser.add_argument_group("clip opts")
    group.add_argument("--finetune_clip", default=False, action="store_true")

    group.add_argument("--freeze_sender_embeddings", default=False, action="store_true")
    group.add_argument("--freeze_recv_embeddings", default=False, action="store_true")
    group.add_argument("--share_embeddings", default=False, action="store_true")

    group.add_argument("--max_clip_vocab", type=int, default=None)


def get_game_opts(parser):
    group = parser.add_argument_group("game opts")
    # multi symbol RNN options
    group.add_argument(
        "--sender_rnn_embed_dim",
        type=int,
        default=512,
    )
    group.add_argument(
        "--sender_rnn_hidden_size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--sender_cell",
        choices=["rnn", "lstm", "gru"],
        default="gru",
        help="Type of the cell used for Sender {rnn, gru, lstm} (default: rnn)",
    )
    parser.add_argument(
        "--informed_sender",
        action="store_true",
        default=False,
    )


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
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="Enable wandb logging",
    )
    parser.add_argument("--wandb_tag", default="default")
    parser.add_argument("--wandb_project", default="playground")

    get_data_opts(parser)
    get_vision_model_opts(parser)
    get_game_opts(parser)
    get_clip_opts(parser)

    opts = core.init(arg_parser=parser, params=params)
    return opts
