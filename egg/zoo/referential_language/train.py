# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import egg.core as core
from egg.zoo.referential_language.data import get_dataloader
from egg.zoo.referential_language.games import build_game
from egg.zoo.referential_language.utils import get_common_opts


def main(params):
    opts = get_common_opts(params=params)
    print(opts)

    if not opts.distributed_context.is_distributed and opts.debug:
        breakpoint()

    data_kwargs = {
        "dataset_dir": opts.dataset_dir,
        "batch_size": opts.batch_size,
        "image_size": opts.image_size,
        "seed": opts.random_seed,
    }
    train_loader = get_dataloader(**data_kwargs)

    game = build_game(opts)

    optimizer = core.build_optimizer(game.parameters())
    callbacks = [core.ConsoleLogger(as_json=True, print_train_loss=True)]

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        callbacks=callbacks,
    )
    trainer.train(n_epochs=opts.n_epochs)
    print("| FINISHED JOB")


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
