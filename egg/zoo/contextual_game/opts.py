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
    group.add_argument("--metadata_dir", default="/private/home/rdessi/imagecode/data")
    group.add_argument("--num_workers", type=int, default=8)


def get_vision_model_opts(parser):
    group = parser.add_argument_group("vision model options")
    group.add_argument(
        "--vision_model",
        type=str,
        default="clip_vit_b/16",
        choices=[
            "resnet50",
            "resnet101",
            "resnet152",
            "vgg11",
            "clip_vit_b/32",
            "clip_vit_b/16",
            "clip_vit_l/14",
            "clip_resnet50",
            "clip_resnet101",
        ],
        help="Model name for the visual encoder",
    )
    group.add_argument(
        "--freeze_vision",
        default=None,
        choices=["sender_only", "recv_only", "both"],
        help="If set, pretrained vision modules will be used",
    )


def get_clip_opts(parser):
    group = parser.add_argument_group("clip opts")
    group.add_argument(
        "--add_clip_tokens",
        default=False,
        action="store_true",
        help="Add clip special tokens for sos and eos to each emergent message",
    )
    group.add_argument(
        "--finetune_clip",
        default=False,
        action="store_true",
        help="Update clip weights during the referential game",
    )
    group.add_argument(
        "--freeze_clip_embeddings",
        default=False,
        action="store_true",
        help="Freeze pretrained clip embeddings during the referential game",
    )
    group.add_argument(
        "--max_clip_vocab",
        type=int,
        default=None,
        help="Max num di embeddings to use from clip",
    )


def get_game_opts(parser):
    group = parser.add_argument_group("game opts")
    group.add_argument(
        "--recv_hidden_dim",
        type=int,
        default=2048,
    )
    group.add_argument(
        "--recv_output_dim",
        type=int,
        default=2048,
    )
    group.add_argument(
        "--clip_receiver",
        default=False,
        action="store_true",
        help="Use a CLIP model as receiver",
    )
    group.add_argument(
        "--sender_clip_embeddings",
        default=False,
        action="store_true",
        help="Use clip embeddings as symbol representations in the sender architecture",
    )
    group.add_argument(
        "--recv_clip_embeddings",
        default=False,
        action="store_true",
        help="Use clip embeddings as symbol representations in the receiver architecture",
    )
    group.add_argument(
        "--use_mlp_recv",
        default=False,
        action="store_true",
        help="Use an mlp in the recv arch when transforming the extracted visual feats (if fale nn.Identity is used)",
    )

    # multi symbol RNN options
    group.add_argument(
        "--sender_rnn_embed_dim",
        type=int,
        default=512,
    )
    group.add_argument(
        "--recv_rnn_embed_dim",
        type=int,
        default=512,
    )
    group.add_argument(
        "--sender_rnn_hidden_size",
        type=int,
        default=512,
    )
    group.add_argument(
        "--recv_rnn_hidden_size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--sender_cell",
        choices=["rnn", "lstm", "gru"],
        default="rnn",
        help="Type of the cell used for Sender {rnn, gru, lstm} (default: rnn)",
    )
    parser.add_argument(
        "--recv_cell",
        default="rnn",
        choices=["rnn", "lstm", "gru"],
        help="Type of the cell used for Receiver {rnn, gru, lstm} (default: rnn)",
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
        "--weight_decay",
        type=float,
        default=10e-6,
        help="Weight decay used for SGD",
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
    parser.add_argument("--wandb_tag", default=None)

    get_data_opts(parser)
    get_vision_model_opts(parser)
    get_game_opts(parser)
    get_clip_opts(parser)

    opts = core.init(arg_parser=parser, params=params)
    return opts
