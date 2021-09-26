# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
from pathlib import Path

import torch
import wandb

import egg.core as core
from egg.zoo.referential_language.callbacks import get_callbacks, MyWandbLogger
from egg.zoo.referential_language.data import get_dataloader
from egg.zoo.referential_language.games import build_game
from egg.zoo.referential_language.utils import add_weight_decay, get_common_opts


def main(params):
    opts = get_common_opts(params=params)
    print(f"{opts}")

    if not opts.distributed_context.is_distributed and opts.pdb:
        breakpoint()

    data_kwargs = {
        "dataset_dir": "/datasets01/open_images/030119",
        "batch_size": opts.batch_size,
        "num_workers": opts.num_workers,
        "contextual_distractors": opts.contextual_distractors,
        "image_size": opts.image_size,
        "is_distributed": opts.distributed_context.is_distributed,
        "seed": opts.random_seed,
    }
    train_loader = get_dataloader(
        split="train", use_augmentations=opts.use_augmentations, **data_kwargs
    )
    validation_loader = get_dataloader(
        split="validation", use_augmentations=False, shuffle=False, **data_kwargs
    )
    test_loader = get_dataloader(
        split="test", use_augmentations=False, shuffle=False, **data_kwargs
    )

    game = build_game(opts)

    model_parameters = add_weight_decay(game, opts.weight_decay, skip_name="bn")

    optimizer = torch.optim.SGD(
        model_parameters,
        lr=opts.lr,
        momentum=0.9,
    )
    optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=opts.n_epochs
    )

    callbacks = get_callbacks(opts)
    if opts.wandb and opts.distributed_context.is_leader:
        callbacks.append(
            MyWandbLogger(
                opts=opts, project="contexualized_emcomm", tags=[opts.wandb_tag]
            )
        )

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        optimizer_scheduler=optimizer_scheduler,
        train_data=train_loader,
        validation_data=validation_loader,
        callbacks=callbacks,
    )
    trainer.train(n_epochs=opts.n_epochs)

    print("| STARTING TEST")
    _, interaction = trainer.eval(test_loader)
    if opts.wandb:
        wandb.log({"test_accuracy": interaction.aux["acc"].mean().item()}, commit=True)
    if opts.checkpoint_dir:
        torch.save(interaction, Path(opts.checkpoint_dir) / "test_interaction")

    print("| FINISHED JOB")


if __name__ == "__main__":
    main(sys.argv[1:])
