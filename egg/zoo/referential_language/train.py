# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import uuid
from pathlib import Path

import torch
import wandb

import egg.core as core
import egg.zoo.referential_language.data_utils as data
from egg.zoo.referential_language.callbacks import get_callbacks
from egg.zoo.referential_language.games import build_game

# from egg.zoo.referential_language.scripts.analyze_interaction import analyze_interaction
from egg.zoo.referential_language.utils import get_common_opts


def get_job_and_task_id(opts):
    job_id = os.environ.get("SLURM_ARRAY_JOB_ID", None)
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", None)
    if job_id is None or task_id is None:
        job_id = os.environ.get("SLURM_JOB_ID", uuid.uuid4())
        task_id = os.environ.get("SLURM_PROCID", 0)

    opts.job_id = job_id
    opts.task_id = task_id
    return job_id, task_id


def main(params):
    opts = get_common_opts(params=params)
    job_id, task_id = get_job_and_task_id(opts)
    print(opts)

    if not opts.distributed_context.is_distributed and opts.debug:
        breakpoint()

    if opts.wandb and opts.distributed_context.is_leader:
        opts.wandb_id = f"{job_id}_{task_id}"
        wandb.init(
            project="contexualized_emcomm",
            id=f"{job_id}_{task_id}",
            tags=[
                f"ctx_dist={opts.contextual_distractors}",
                opts.attention_type,
                opts.context_integration,
                f"pretrain={opts.pretrain_vision}",
                f"2_layers={opts.two_layers_recv}",
            ],
        )
        wandb.config.update(opts)

    data_kwargs = {
        "image_dir": opts.image_dir,
        "metadata_dir": opts.metadata_dir,
        "split": "train",
        "batch_size": opts.batch_size,
        "image_size": opts.image_size,
        "max_objects": opts.max_objects,
        "contextual_distractors": opts.contextual_distractors,
        "use_augmentations": opts.use_augmentations,
        "is_distributed": opts.distributed_context.is_distributed,
        "seed": opts.random_seed,
    }
    train_loader = data.get_dataloader(**data_kwargs)

    game = build_game(opts)

    optimizer = core.build_optimizer(game.parameters())

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        callbacks=get_callbacks(opts),
    )
    # trainer.train(n_epochs=opts.n_epochs)

    # VALIDATION LOOP
    def log_wandb(value, name):
        if opts.wandb and opts.distributed_context.is_leader:
            wandb.log({name: value})

    def log_stats(interaction, mode):
        dump = dict((k, v.mean().item()) for k, v in interaction.aux.items())
        dump.update(dict(mode=mode))
        print(json.dumps(dump), flush=True)

    def dump_interaction(interaction):
        if opts.checkpoint_dir and opts.distributed_context.is_leader:
            output_path = Path(opts.checkpoint_dir) / "interactions"
            output_path.mkdir(exist_ok=True, parents=True)
            interaction_name = f"val_interaction_{job_id}_{task_id}"
            torch.save(interaction, output_path / interaction_name)

    def process_interaction(loader_kwargs, log_name):
        val_loader = data.get_dataloader(**loader_kwargs)
        _, val_interaction = trainer.eval(val_loader)
        val_interaction.aux_input.update({"args": opts})
        log_stats(val_interaction, log_name)
        log_wandb(val_interaction.aux["acc"].mean().item(), "val_acc")
        # dump_interaction(val_interaction, interaction_name)
        # analyze_interaction(val_interaction)

    val_data_kwargs = dict(data_kwargs)
    val_data_kwargs.update({"split": "val", "use_augmentations": False})

    process_interaction(val_data_kwargs, "VALIDATION SET")

    # GAUSSIAN TEST
    gaussian_data = data.get_gaussian_dataloader(**data_kwargs)
    _, gaussian_interaction = trainer.eval(gaussian_data)
    log_stats(gaussian_interaction, "GAUSSIAN TEST")
    log_wandb(gaussian_interaction.aux["acc"].mean(), "Gaussian acc")

    print("| FINISHED JOB")


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
