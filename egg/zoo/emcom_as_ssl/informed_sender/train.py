# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

import egg.core as core
from egg.zoo.emcom_as_ssl.informed_sender.data import get_dataloader
from egg.zoo.emcom_as_ssl.callbacks import add_wandb_logger, get_callbacks
from egg.zoo.emcom_as_ssl.informed_sender.games import build_game
from egg.zoo.emcom_as_ssl.informed_sender.utils import get_game_opts
from egg.zoo.emcom_as_ssl.LARC import LARC
from egg.zoo.emcom_as_ssl.utils import add_weight_decay, get_common_opts


def main(params):
    parser = get_common_opts()
    parser = get_game_opts(parser)
    opts = core.init(arg_parser=parser, params=params)
    assert opts.batch_size % opts.game_size == 0
    print(f"{opts}\n")

    if not opts.distributed_context.is_distributed and opts.pdb:
        breakpoint()

    train_loader = get_dataloader(
        dataset_dir=opts.dataset_dir,
        image_size=opts.image_size,
        batch_size=opts.batch_size,
        game_size=opts.game_size,
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

    callbacks = get_callbacks(checkpoint_freq=opts.checkpoint_freq)
    if opts.wandb:
        add_wandb_logger(callbacks, opts, game)

    callbacks.append(core.EarlyStopperAccuracy(0.9999, validation=False))
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
