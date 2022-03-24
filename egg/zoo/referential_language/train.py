# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time

import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR

import egg.core as core
from egg.core.batch import Batch
from egg.core.interaction import Interaction, LoggingStrategy
from egg.zoo.referential_language.dataloaders import get_dataloader
from egg.zoo.referential_language.game import build_game
from egg.zoo.referential_language.utils.callbacks import get_callbacks
from egg.zoo.referential_language.utils.opts import get_common_opts
from egg.zoo.referential_language.utils.helpers import (
    dump_interaction,
    get_sha,
    log_stats,
    setup_for_distributed,
    store_job_and_task_id,
)


def test(game, data_kwargs):
    ds_name = data_kwargs["dataset_name"]
    data_kwargs.update({"dataset_name": "gaussian"})
    gaussian_dataloader = get_dataloader(**data_kwargs)
    gaussian_interaction = test_loop(game, gaussian_dataloader)
    log_stats(gaussian_interaction, "GAUSSIAN SET")

    game = game.module if dist.is_initialized() else game
    logging_test_args = [False, False, True, True, True, True, False]
    game.test_logging_strategy = LoggingStrategy(*logging_test_args)

    data_kwargs.update({"split": "test", "dataset_name": ds_name})
    test_dataloader = get_dataloader(**data_kwargs)
    test_interaction = test_loop(game, test_dataloader)
    log_stats(test_interaction, "TEST SET")

    return test_interaction


def test_loop(game, data, device=None):
    game.eval()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    interactions = []
    with torch.no_grad():
        for batch_id, batch in enumerate(data):
            if not isinstance(batch, Batch):
                batch = Batch(*batch)
            batch = batch.to(device)

            _, interaction = game(*batch)

            if dist.is_initialized():
                interaction = Interaction.gather_distributed_interactions(interaction)
            interactions.append(interaction.to("cpu"))

    return Interaction.from_iterable(interactions)


def main(params):
    start = time.time()
    opts = get_common_opts(params=params)

    store_job_and_task_id(opts)
    setup_for_distributed(opts.distributed_context.is_leader)
    print(opts)
    print(get_sha())

    data_kwargs = {
        "dataset_name": opts.dataset_name,
        "image_dir": opts.image_dir,
        "metadata_dir": opts.metadata_dir,
        "batch_size": opts.batch_size,
        "split": "train",
        "image_size": opts.image_size,
        "max_objects": opts.max_objects,
        "use_augmentation": opts.use_augmentation,
        "seed": opts.random_seed,
    }

    train_loader = get_dataloader(**data_kwargs)

    data_kwargs.update({"split": "val"})
    val_loader = get_dataloader(**data_kwargs)

    game = build_game(opts)
    optimizer = torch.optim.Adam(game.parameters(), lr=opts.lr)
    optimizer_scheduler = CosineAnnealingLR(optimizer, T_max=opts.n_epochs)

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        optimizer_scheduler=optimizer_scheduler,
        train_data=train_loader,
        validation_data=val_loader,
        callbacks=get_callbacks(opts),
        debug=opts.debug,
    )
    trainer.train(n_epochs=opts.n_epochs)

    test_interaction = test(trainer.game, data_kwargs)
    if opts.distributed_context.is_leader:
        dump_interaction(test_interaction, opts)
    end = time.time()
    print(f"| Run took {end - start:.2f} seconds")
    print("| FINISHED JOB")


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
