# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interaction_path", help="Run the game with pdb enabled")
    return parser.parse_args()


def main():
    get_opts()


if __name__ == "__main__":
    main()
