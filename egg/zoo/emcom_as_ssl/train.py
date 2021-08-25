# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import wandb

import egg.core as core
from egg.zoo.emcom_as_ssl.utils import add_weight_decay, get_common_opts
from egg.zoo.emcom_as_ssl.data import get_dataloader
from egg.zoo.emcom_as_ssl.games import build_game
from egg.zoo.emcom_as_ssl.game_callbacks import get_callbacks, WandbLogger
from egg.zoo.emcom_as_ssl.LARC import LARC


def main(params):
    opts = get_common_opts(params=params)
    print(f"{opts}\n")
    assert not opts.batch_size % 2, "Batch size must be multiple of 2"

    if not opts.distributed_context.is_distributed and opts.pdb:
        breakpoint()

    train_loader = get_dataloader(
        dataset_dir=opts.dataset_dir,
        image_size=opts.image_size,
        batch_size=opts.batch_size,
        num_workers=opts.num_workers,
        use_augmentations=opts.use_augmentations,
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

    if opts.use_larc:
        optimizer = LARC(optimizer, trust_coefficient=0.001, clip=False, eps=1e-8)

    callbacks = get_callbacks()
    if opts.wandb and opts.distributed_context.is_leader:
        run_name = opts.checkpoint_dir.split("/")[-1] if opts.checkpoint_dir else ""
        wandb.init(
            project="post_rebuttal", id=run_name, resume=True, tags=[opts.wandb_tag]
        )
        wandb.config.update(opts)
        wandb.watch(game, log="all")

        callbacks.append(WandbLogger())

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        optimizer_scheduler=optimizer_scheduler,
        train_data=train_loader,
        callbacks=callbacks,
    )
    trainer.train(n_epochs=opts.n_epochs)

    print("| FINISHED JOB")


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    import sys

    main(sys.argv[1:])
