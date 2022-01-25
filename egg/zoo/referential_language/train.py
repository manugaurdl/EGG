# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import uuid

import egg.core as core
from egg.zoo.referential_language.callbacks import get_callbacks
from egg.zoo.referential_language.data import get_dataloader
from egg.zoo.referential_language.eval import run_gaussian_test, run_test
from egg.zoo.referential_language.games import build_game
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
    start = time.time()
    opts = get_common_opts(params=params)
    job_id, task_id = get_job_and_task_id(opts)
    print(opts)

    data_kwargs = {
        "image_dir": opts.image_dir,
        "metadata_dir": opts.metadata_dir,
        "batch_size": opts.batch_size,
        "split": "train",
        "image_size": opts.image_size,
        "max_objects": opts.max_objects,
        "seed": opts.random_seed,
    }

    train_loader = get_dataloader(**data_kwargs)
    data_kwargs.update({"split": "val"})
    val_loader = get_dataloader(**data_kwargs)

    game = build_game(opts)
    optimizer = core.build_optimizer(game.parameters())

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data=val_loader,
        callbacks=get_callbacks(opts),
        debug=opts.debug,
    )
    trainer.train(n_epochs=opts.n_epochs)

    data_kwargs.update({"split": "test"})
    data_loader = get_dataloader(**data_kwargs)
    run_test(trainer, opts, data_loader)

    run_gaussian_test(trainer, opts, data_kwargs)

    print("| FINISHED JOB")
    end = time.time()
    print(f"| Run took {end - start:.2f} seconds")


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
