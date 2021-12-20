# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import egg.core as core


def get_data_opts(parser):
    group = parser.add_argument_group("data options")
    group.add_argument(
        "--image_dir",
        type=str,
        default="/datasets01/VisualGenome1.2/061517/",
        help="Path to folder with VG images",
    )
    group.add_argument(
        "--metadata_dir",
        type=str,
        default="/private/home/rdessi/visual_genome",
        help="Path to folder with VG metada",
    )
    group.add_argument(
        "--max_objects",
        type=int,
        default=20,
        help="Max numbeer of bboxes to extract from an image",
    )
    group.add_argument("--image_size", type=int, default=64, help="Image size")
    group.add_argument("--contextual_distractors", action="store_true", default=False)
    group.add_argument("--use_augmentations", action="store_true", default=False)


def get_vision_module_opts(parser):
    group = parser.add_argument_group("vision module options")
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
    group = parser.add_argument_group("game architecture options")
    group.add_argument(
        "--context_integration",
        default="cat",
        choices=["cat", "gate"],
        help="Cat concatenates visual context with object features gate uses context to gate object features",
    )
    group.add_argument(
        "--num_heads",
        type=int,
        default=0,
        help="Number of heads in the self attention to integrate context with objects (default: 0 means no attention)",
    )
    group.add_argument(
        "--recv_temperature",
        type=float,
        default=0.1,
        help="Temperature for similarity computation in the loss fn. Ignored when similarity is 'dot'",
    )
    group.add_argument(
        "--residual",
        action="store_true",
        default=False,
        help="If set, it will add a residual connection when calculating attention with gate ctx integration",
    )
    group.add_argument(
        "--cosine_similarity",
        action="store_true",
        default=False,
        help="If True Receiver will compute l2-normalized dot product between message and images (default: False)",
    )


def get_gs_opts(parser):
    group = parser.add_argument_group("gumbel softmax training options")
    group.add_argument(
        "--gs_temperature",
        type=float,
        default=1.0,
        help="gs temperature used in the relaxation layer",
    )


def get_rf_opts(parser):
    pass
    # group = parser.add_argument_group("reinforce training options")
    # group.add_argument("--", type=float, default=None, help="")


def get_single_symbol_opts(parser):
    group = parser.add_argument_group("single symbol options")
    group.add_argument(
        "--recv_hidden_dim",
        type=int,
        default=2048,
        help="Hidden dim of the non-linear projection of the distractors in the receiver agent",
    )
    group.add_argument(
        "--recv_output_dim",
        type=int,
        default=2048,
        help="Output dim of the non-linear projection of the distractors, used to compare with msg embedding",
    )


def get_multi_symbol_opts(parser):
    group = parser.add_argument_group("multi symbol options")
    group.add_argument(
        "--sender_cell_dim",
        default=256,
        type=int,
        help="Size of sender hidden unit in recurrent network",
    )
    group.add_argument(
        "--recv_cell_dim",
        default=256,
        type=int,
        help="Size of recv hidden unit in recurrent network",
    )
    group.add_argument(
        "--sender_embed_dim",
        default=512,
        type=int,
        help="Size of sender embeddings in recurrent network",
    )
    group.add_argument(
        "--recv_embed_dim",
        default=512,
        type=int,
        help="Size of sender embeddings in recurrent network",
    )
    group.add_argument(
        "--sender_cell",
        default="rnn",
        choices=["rnn", "gru", "lstm"],
        help="Type of recurrent unit for generating a message in sender agent",
    )
    group.add_argument(
        "--recv_cell",
        default="rnn",
        choices=["rnn", "gru", "lstm"],
        help="Type of recurrent unit for generating a message in receiver agent",
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
        "--game_mode",
        default="gs",
        choices=["gs", "rf"],
        help="Choose between gumbel-based and reinforce-based training (default: gs)",
    )

    get_data_opts(parser)
    get_gs_opts(parser)
    get_vision_module_opts(parser)
    get_game_arch_opts(parser)
    get_single_symbol_opts(parser)
    get_multi_symbol_opts(parser)
    get_rf_opts(parser)

    opts = core.init(arg_parser=parser, params=params)
    return opts
