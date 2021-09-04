# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import pathlib
from typing import Union

import torch
import torch.nn as nn

from egg.core.interaction import Interaction
from egg.core.util import move_to

O_TEST_PATH = (
    "/private/home/mbaroni/agentini/representation_learning/"
    "generalizaton_set_construction/80_generalization_data_set/"
)
I_TEST_PATH = "/datasets01/imagenet_full_size/061417/val"


def add_common_cli_args(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
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
    return parser


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
