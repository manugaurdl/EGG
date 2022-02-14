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
        default="/private/home/rdessi/visual_genome",
        help="Path to folder with VG images",
    )
    group.add_argument(
        "--metadata_dir",
        type=str,
        default="/private/home/rdessi/visual_genome/last_version_metadata/filtered_splits",
        help="Path to folder with VG metada",
    )
    group.add_argument(
        "--max_objects",
        type=int,
        default=10,
        help="Max number of bboxes to extract from an image",
    )
    group.add_argument("--image_size", type=int, default=32, help="Image size")
    group.add_argument("--random_distractors", default=False, action="store_true")


def get_vision_module_opts(parser):
    group = parser.add_argument_group("vision module options")
    group.add_argument(
        "--vision_model",
        type=str,
        default="resnet34",
        choices=["resnet18", "resnet34", "resnet50", "resnet101"],
        help="Model name for the encoder",
    )
    group.add_argument(
        "--pretrain_vision",
        default=False,
        action="store_true",
        help="If set, pretrained vision models will be used",
    )


def get_attention_opts(parser):
    group = parser.add_argument_group("sender attention options")
    group.add_argument(
        "--attn_type",
        default="none",
        choices=["none", "dot", "self", "top"],
    )
    group.add_argument(
        "--attn_topk",
        default=1,
        type=int,
        help="Use top k most similar distractors to compute the attention vector",
    )
    group.add_argument(
        "--num_heads",
        default=1,
        type=int,
        help="Number of attention heads used in self attention",
    )
    group.add_argument(
        "--random_k",
        default=False,
        action="store_true",
        help="Use k random distractors to compute the attention vector",
    )


def get_game_arch_opts(parser):
    group = parser.add_argument_group("game architecture options")
    group.add_argument(
        "--loss_temperature",
        type=float,
        default=1.0,
        help="Temperature for similarity computation in the loss fn. Ignored when similarity is 'dot'",
    )
    group.add_argument(
        "--output_dim",
        type=int,
        default=1024,
        help="Output dim of the non-linear projection of the distractors, used to compare with msg embedding",
    )
    group.add_argument(
        "--recv_separate_embeddings",
        default=False,
        action="store_true",
        help="Use separate embedding matrix to transform tgt message and ctx message when attention is used",
    )
    group.add_argument(
        "--sender_separate_messages",
        default=False,
        action="store_true",
        help="Use separate embedding matrix to generate a tgt message and ctx message when attention is used",
    )
    group.add_argument(
        "--random_ctx_position",
        default=False,
        action="store_true",
        help="Randomly shuffle target and context message when attention is used",
    )


def get_gs_opts(parser):
    group = parser.add_argument_group("gumbel softmax training options")
    group.add_argument(
        "--gs_temperature",
        type=float,
        default=1.0,
        help="gs temperature used in the relaxation layer",
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
    get_attention_opts(parser)

    opts = core.init(arg_parser=parser, params=params)
    setup_for_distributed(opts.distributed_context.is_leader)
    return opts


def setup_for_distributed(is_master):
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
