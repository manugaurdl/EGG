# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

import egg.core as core
from egg.core import EarlyStopperAccuracy
from egg.zoo.simclr.contrastive.data import get_dataloader
from egg.zoo.simclr.contrastive.games import build_game
from egg.zoo.simclr.contrastive.game_callbacks import (
    BestStatsTracker,
    DistributedSamplerEpochSetter,
    VisionModelSaver
)
from egg.zoo.simclr.contrastive.utils import add_weight_decay, get_opts


def main(params):
    opts = get_opts(params=params)
    print(opts)
    print(
        f"Running a distruted training is set to: {opts.distributed_context.is_distributed}. "
        f"World size is {opts.distributed_context.world_size}\n"
        f"Using imagenet with image size: {opts.image_size}. "
        f"Using batch of size {opts.batch_size} on {opts.distributed_context.world_size} device(s)"
    )
    if not opts.distributed_context.is_distributed and opts.pdb:
        breakpoint()

    train_loader = get_dataloader(
        dataset_dir=opts.dataset_dir,
        image_size=opts.image_size,
        batch_size=opts.batch_size,
        num_workers=opts.num_workers,
        is_distributed=opts.distributed_context.is_distributed,
        seed=opts.random_seed
    )

    simclr_game = build_game(opts)

    model_parameters = add_weight_decay(
        simclr_game,
        opts.weight_decay,
        skip_name='bn'
    )

    optimizer = torch.optim.SGD(
        model_parameters,
        lr=opts.lr,
        momentum=0.9,
    )
    optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opts.n_epochs)

    callbacks = [
        core.ConsoleLogger(as_json=True, print_train_loss=True),
        EarlyStopperAccuracy(opts.early_stopping_thr, validation=False),
        BestStatsTracker(),
        VisionModelSaver()
    ]

    if opts.distributed_context.is_distributed:
        callbacks.append(DistributedSamplerEpochSetter())

    trainer = core.Trainer(
        game=simclr_game,
        optimizer=optimizer,
        optimizer_scheduler=optimizer_scheduler,
        train_data=train_loader,
        callbacks=callbacks
    )
    trainer.train(n_epochs=opts.n_epochs)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
