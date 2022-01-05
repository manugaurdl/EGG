# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from pathlib import Path

import torch
import wandb

import egg.zoo.referential_language.data_utils as data


def log_stats(interaction, mode):
    dump = dict((k, v.mean().item()) for k, v in interaction.aux.items())
    dump.update(dict(mode=mode))
    print(json.dumps(dump), flush=True)


def perform_gaussian_test(trainer, data_kwargs):
    gaussian_data = data.get_gaussian_dataloader(**data_kwargs)
    _, gaussian_interaction = trainer.eval(gaussian_data)
    log_stats(gaussian_interaction, "GAUSSIAN TEST")


def run_evaluation_loop(trainer, opts, data_kwargs):
    val_data_kwargs = dict(data_kwargs)
    val_data_kwargs.update({"split": "val", "use_augmentations": False})
    val_loader = data.get_dataloader(**val_data_kwargs)
    _, val_interaction = trainer.eval(val_loader)
    val_interaction.aux_input.update({"args": opts})
    log_stats(val_interaction, "VALIDATION SET")

    if opts.distributed_context.is_leader:
        if opts.checkpoint_dir:
            output_path = Path(opts.checkpoint_dir) / "interactions"
            output_path.mkdir(exist_ok=True, parents=True)
            interaction_name = f"val_interaction_{opts.job_id}_{opts.task_id}"
            torch.save(val_interaction, output_path / interaction_name)
        if opts.wandb:
            wandb.log({"val_acc": val_interaction.aux["acc"].mean().item()})
