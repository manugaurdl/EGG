# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse


def get_opts(parser):
    group = parser.add_argument_group("game architecture")
    group.add_argument(
        "--game_size",
        type=int,
        default=2,
        help="Image candidates lineup for the communication game",
    )


def get_game_opts(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    get_opts(parser)

    return parser
