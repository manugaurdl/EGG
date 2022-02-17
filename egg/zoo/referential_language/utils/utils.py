# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import subprocess

import egg.core as core
from egg.zoo.referential_language.utils.opts import (
    get_attention_opts,
    get_data_opts,
    get_game_arch_opts,
    get_gs_opts,
    get_vision_module_opts,
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


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    sha = "N/A"
    diff = "clean"
    branch = "N/A"
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        subprocess.check_output(["git", "diff"], cwd=cwd)
        diff = _run(["git", "diff-index", "HEAD"])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message
