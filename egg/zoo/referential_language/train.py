# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from pathlib import Path

import torch

import egg.core as core
from egg.zoo.referential_language.callbacks import get_callbacks
from egg.zoo.referential_language.data import get_dataloader
from egg.zoo.referential_language.games import build_game
from egg.zoo.referential_language.utils import get_common_opts


def main(params):
    opts = get_common_opts(params=params)
    print(opts)

    if not opts.distributed_context.is_distributed and opts.debug:
        breakpoint()

    data_kwargs = {
        "image_dir": opts.image_dir,
        "metadata_dir": opts.metadata_dir,
        "split": "train",
        "batch_size": opts.batch_size,
        "image_size": opts.image_size,
        "max_objects": opts.max_objects,
        "contextual_distractors": opts.contextual_distractors,
        "seed": opts.random_seed,
    }
    train_loader = get_dataloader(**data_kwargs)

    game = build_game(opts)

    optimizer = core.build_optimizer(game.parameters())

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        callbacks=get_callbacks(),
    )
    trainer.train(n_epochs=opts.n_epochs)

    val_data_kwargs = dict(data_kwargs)
    val_loader = get_dataloader(**val_data_kwargs.update({"split": "val"}))
    _, val_interaction = trainer.eval(val_loader)

    dump = dict((k, v.mean().item()) for k, v in val_interaction.aux.items())
    dump.update(dict(mode="VALIDATION_SET"))
    print(json.dumps(dump), flush=True)

    if opts.checkpoint_dir:
        output_path = Path(opts.checkpoint_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        torch.save(val_interaction, output_path / "val_interaction")

    print("| FINISHED JOB")


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
