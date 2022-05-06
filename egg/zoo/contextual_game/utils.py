# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import subprocess
import uuid
from pathlib import Path

import torch

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
        "--pretrain_vision",
        default=False,
        action="store_true",
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
        "--use_clip_embeddings",
        default=False,
        action="store_true",
        help="Use clip embeddings as symbol representations",
    )
    group.add_argument(
        "--use_mlp_recv",
        default=False,
        action="store_true",
        help="Use an mlp in the recv arch when transforming the extracted visual feats (if fale nn.Identity is used)",
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

    get_data_opts(parser)
    get_vision_model_opts(parser)
    get_game_opts(parser)
    get_clip_opts(parser)

    opts = core.init(arg_parser=parser, params=params)
    return opts


def store_job_and_task_id(opts):
    job_id = os.environ.get("SLURM_ARRAY_JOB_ID", None)
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", None)
    if job_id is None or task_id is None:
        job_id = os.environ.get("SLURM_JOB_ID", uuid.uuid4())
        task_id = os.environ.get("SLURM_PROCID", 0)

    opts.job_id = job_id
    opts.task_id = task_id
    return job_id, task_id


def add_weight_decay(model, weight_decay=1e-5, skip_name=""):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or skip_name in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


def log_stats(interaction, mode):
    dump = dict((k, v.mean().item()) for k, v in interaction.aux.items())
    dump.update(dict(mode=mode))
    print(json.dumps(dump), flush=True)


def dump_interaction(interaction, opts):
    if opts.checkpoint_dir:
        output_path = Path(opts.checkpoint_dir) / "interactions"
        output_path.mkdir(exist_ok=True, parents=True)
        interaction_name = f"interaction_{opts.job_id}_{opts.task_id}"

        interaction.aux_input["args"] = opts
        torch.save(interaction, output_path / interaction_name)


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
