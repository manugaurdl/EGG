# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import pathlib
import random
from typing import Union

import numpy as np
import torch
import torch.nn as nn

from egg.core.interaction import Interaction
from egg.core.util import move_to
from egg.zoo.emcom_as_ssl.games import build_game


def add_common_cli_args(parser):
    parser.add_argument("--checkpoint_path", type=str, help="Path to model checkpoint")
    parser.add_argument(
        "--shared_vision",
        default=False,
        action="store_true",
        help="Load a model with shared vision module",
    )
    parser.add_argument(
        "--pretrain_vision",
        default=False,
        action="store_true",
        help="Load a model with a pretrained shared vision module",
    )
    parser.add_argument("--vocab_size", default=2048, type=int, help="Vocabulary size")
    parser.add_argument("--batch_size", default=128, type=int, help="Test batch size")
    parser.add_argument(
        "--evaluate_with_augmentations",
        default=False,
        action="store_true",
        help="Running gaussian evaluation with data augmentation",
    )
    parser.add_argument(
        "--test_set",
        type=str,
        choices=["o_test", "i_test"],
        default="o_test",
        help="Choose which imagenet validation test to use, choices [i_test, o_test] (default: o_test)",
    )
    parser.add_argument(
        "--dump_interaction_folder",
        type=str,
        default=None,
        help="Path where interaction will be saved. If None or empty string interaction won't be saved",
    )
    parser.add_argument(
        "--pdb", default=False, action="store_true", help="Run with pdb"
    )


def get_params(
    shared_vision: bool,
    pretrain_vision: bool,
    vocab_size: int,
):
    assert not pretrain_vision or shared_vision
    params = dict(
        shared_vision=shared_vision,
        pretrain_vision=pretrain_vision,
        vocab_size=vocab_size,
    )

    params_fixed = dict(
        informed_sender=False,  # TODO INFORMED SENDER
        random_seed=111,
        model_name="resnet50",
        gs_temperature=1.0,
        similarity_temperature=0.1,
        output_dim=2048,
        image_size=224,
        distributed_context=argparse.Namespace(is_distributed=False),
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


def save_interaction(
    interaction: Interaction,
    log_dir: Union[pathlib.Path, str],
    test_set: str,
):
    dump_dir = pathlib.Path(log_dir)
    dump_dir.mkdir(exist_ok=True, parents=True)
    torch.save(interaction, dump_dir / f"interactions_{test_set}.pt")


def evaluate(
    game: nn.Module,
    data: torch.utils.data.DataLoader,
):
    if torch.cuda.is_available():
        game.cuda()
    game.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean_loss, mean_accuracy = 0.0, 0.0
    interactions = []
    n_batches = 0
    with torch.no_grad():
        for batch in data:
            batch = move_to(batch, device)
            optimized_loss, interaction = game(*batch)

            interaction = interaction.to("cpu")
            interactions.append(interaction)

            mean_loss += optimized_loss
            mean_accuracy += interaction.aux["acc"].mean().item()
            n_batches += 1
            if n_batches % 10 == 0:
                print(f"finished batch {n_batches}")

    print(f"processed {n_batches} batches in total")
    mean_loss /= n_batches
    mean_accuracy /= n_batches
    full_interaction = Interaction.from_iterable(interactions)

    return mean_loss, mean_accuracy, full_interaction
