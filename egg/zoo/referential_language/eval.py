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
    data_loader = data.get_gaussian_dataloader(**data_kwargs)
    data.gaussian_eval(trainer.game, data_loader, trainer.device)


def run_evaluation_loop(trainer, opts, data_kwargs):
    data_kwargs.update({"split": "test"})
    data_loader = data.get_dataloader(**data_kwargs)

    _, interaction = trainer.eval(data_loader)
    log_stats(interaction, "TEST SET")

    if opts.distributed_context.is_leader and opts.checkpoint_dir:
        output_path = Path(opts.checkpoint_dir) / "interactions"
        output_path.mkdir(exist_ok=True, parents=True)
        interaction_name = f"interaction_{opts.job_id}_{opts.task_id}"

        if interaction.aux_input:
            interaction.aux_input.update({"args": opts})
        else:
            interaction.aux_input = {"args": opts}
        torch.save(interaction, output_path / interaction_name)
    if opts.wandb:
        wandb.log({"test_acc": interaction.aux["acc"].mean().item()})
