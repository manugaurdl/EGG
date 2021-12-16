# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from pathlib import Path

import torch

import egg.core as core
import egg.zoo.referential_language.data_utils as data
from egg.zoo.referential_language.callbacks import get_callbacks
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
        callbacks=get_callbacks(),
    )
    trainer.train(n_epochs=opts.n_epochs)

    # VALIDATION LOOP
    def log_stats(interaction, mode):
        dump = dict((k, v.mean().item()) for k, v in interaction.aux.items())
        dump.update(dict(mode=mode))
        print(json.dumps(dump), flush=True)

    val_data_kwargs = dict(data_kwargs)
    val_data_kwargs.update({"split": "val", "use_augmentations": False})
    val_loader = data.get_dataloader(**val_data_kwargs)
    _, val_interaction = trainer.eval(val_loader)
    val_interaction.aux_input.update({"args": val_data_kwargs})
    log_stats(val_interaction, "VALIDATION SET")

    swapped_val_data_kwargs = dict(val_data_kwargs)
    swapped_ctx_distractors = not opts.contextual_distractors
    swapped_val_data_kwargs.update({"contextual_distractors": swapped_ctx_distractors})
    swapped_val_loader = data.get_dataloader(**swapped_val_data_kwargs)
    _, swapped_val_interaction = trainer.eval(swapped_val_loader)
    swapped_val_interaction.aux_input.update({"args": swapped_val_data_kwargs})
    log_stats(swapped_val_interaction, "SWAPPED VALIDATION SET")

    if opts.checkpoint_dir and opts.distributed_context.is_leader:
        output_path = Path(opts.checkpoint_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        interaction_name = (
            f"val_interaction_voc_{opts.vocab_size}_attn_{opts.attention}_"
            f"ctx_integration_{opts.context_integration}_bsz_{opts.batch_size}"
        )
        torch.save(val_interaction, output_path / interaction_name)
        torch.save(swapped_val_interaction, output_path / f"{interaction_name}_swapped")

    # GAUSSIAN TEST
    gaussian_data = data.get_gaussian_dataloader(**data_kwargs)
    _, gaussian_interaction = trainer.eval(gaussian_data)
    log_stats(gaussian_interaction, "GAUSSIAN TEST")

    print("| FINISHED JOB")


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
