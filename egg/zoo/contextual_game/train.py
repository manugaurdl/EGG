# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time

import torch

import egg.core as core
from egg.core import ConsoleLogger
from egg.core.callbacks import WandbLogger

from egg.zoo.contextual_game.data import get_dataloader
from egg.zoo.contextual_game.callbacks import (
    BestStatsTracker,
    DistributedSamplerEpochSetter,
)
from egg.zoo.contextual_game.game import build_game
from egg.zoo.contextual_game.opts import get_common_opts
from egg.zoo.contextual_game.utils import (
    add_weight_decay,
    dump_interaction,
    get_sha,
    log_stats,
    setup_for_distributed,
    store_job_and_task_id,
)


def main(params):
    start = time.time()
    opts = get_common_opts(params=params)

    store_job_and_task_id(opts)
    setup_for_distributed(opts.distributed_context.is_leader)
    print(opts)
    print(get_sha())

    if not opts.distributed_context.is_distributed and opts.debug:
        breakpoint()

    train_loader = get_dataloader(
        image_dir=opts.image_dir,
        metadata_dir=opts.metadata_dir,
        batch_size=opts.batch_size,
        image_size=opts.image_size,
        split="train",
        num_workers=opts.num_workers,
        is_distributed=opts.distributed_context.is_distributed,
        seed=opts.random_seed,
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

    callbacks = [
        ConsoleLogger(as_json=True, print_train_loss=True),
        BestStatsTracker(),
        WandbLogger(opts=opts, project="contexualized_emcomm"),
    ]

    if opts.distributed_context.is_distributed:
        callbacks.append(DistributedSamplerEpochSetter())

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        optimizer_scheduler=optimizer_scheduler,
        train_data=train_loader,
        callbacks=callbacks,
        debug=opts.debug,
    )
    if not opts.eval_only:
        trainer.train(n_epochs=opts.n_epochs)

    # TEST
    test_loader = get_dataloader(
        image_dir=opts.image_dir,
        metadata_dir=opts.metadata_dir,
        batch_size=opts.batch_size,
        image_size=opts.image_size,
        split="valid",
        num_workers=opts.num_workers,
        is_distributed=opts.distributed_context.is_distributed,
        seed=opts.random_seed,
    )

    _, test_interaction = trainer.eval(test_loader)
    log_stats(test_interaction, "TEST SET")
    if opts.distributed_context.is_leader:
        dump_interaction(test_interaction, opts)

    end = time.time()
    print(f"| Run took {end - start:.2f} seconds")
    print("| FINISHED JOB")


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    torch.set_deterministic(True)
    import sys

    main(sys.argv[1:])
