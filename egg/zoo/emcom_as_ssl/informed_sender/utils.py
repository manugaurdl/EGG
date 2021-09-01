# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse


def get_informed_sender_opts(parser):
    group = parser.add_argument_group("game architecture")
    group.add_argument(
        "--informed_sender",
        default=False,
        action="store_true",
        help="If set, Sender will be the one from Lazaridou et al 2017",
    )
    group.add_argument(
        "--force_compare_two",
        default=False,
        action="store_true",
    )


def get_game_opts(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    get_informed_sender_opts(parser)

    return parser
