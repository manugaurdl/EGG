# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import random

import numpy as np
import torch

from egg.zoo.emcom_as_ssl.informed_sender.games import build_game


def add_eval_opts(parser):
    group = parser.add_argument_group("game architecture")
    group.add_argument(
        "--game_size",
        type=int,
        default=2,
        help="image candidates lineup for the communication game",
    )
    group.add_argument("--force_compare_two", default=False, action="store_true")


def get_params(
    shared_vision: bool,
    pretrain_vision: bool,
    vocab_size: int,
    game_size: int,
    batch_size: int = 128,
    **other_params,
):
    print(f"Batch_size is {batch_size}")
    assert not pretrain_vision or shared_vision
    params = dict(
        shared_vision=shared_vision,
        pretrain_vision=pretrain_vision,
        vocab_size=vocab_size,
        game_size=game_size,
        batch_size=batch_size,
    )

    params_fixed = dict(
        random_seed=111,
        model_name="resnet50",
        gs_temperature=1.0,
        similarity_temperature=0.1,
        output_dim=2048,
        image_size=224,
        distributed_context=argparse.Namespace(is_distributed=False),
        **other_params,
    )
    params.update(params_fixed)
    params = argparse.Namespace(**params)

    random.seed(params.random_seed)
    torch.manual_seed(params.random_seed)
    np.random.seed(params.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(params.random_seed)
    return params


def get_game(params: argparse.Namespace, checkpoint_path: str):
    game = build_game(params)
    checkpoint = torch.load(checkpoint_path)
    game.load_state_dict(checkpoint.model_state_dict)
    return game
