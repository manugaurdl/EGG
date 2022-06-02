# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time

import torch
import wandb
from transformers import BartTokenizer

import egg.core as core
from egg.core import ConsoleLogger
from egg.zoo.contextual_game.data import get_dataloader
from egg.zoo.contextual_game.callbacks import WandbLogger
from egg.zoo.contextual_game.game import build_game
from egg.zoo.contextual_game.opts import get_common_opts
from egg.zoo.contextual_game.utils import (
    dump_interaction,
    get_sha,
    log_stats,
    store_job_and_task_id,
)


def main(params):
    start = time.time()
    opts = get_common_opts(params=params)

    store_job_and_task_id(opts)
    print(opts)
    print(get_sha())

    if not opts.distributed_context.is_distributed and opts.debug:
        breakpoint()

    test_loader = get_dataloader(
        image_dir=opts.image_dir,
        metadata_dir=opts.metadata_dir,
        batch_size=opts.batch_size,
        image_size=opts.image_size,
        bart_tokenizer_name=opts.bart_model,
        split="test",
        num_workers=opts.num_workers,
        is_distributed=opts.distributed_context.is_distributed,
        seed=opts.random_seed,
    )

    game = build_game(opts)

    optimizer = torch.optim.Adam(
        game.parameters(), lr=opts.lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2
    )

    callbacks = [ConsoleLogger(as_json=True, print_train_loss=True)]
    if opts.wandb and opts.distributed_context.is_leader:
        callbacks.append(
            WandbLogger(opts=opts, tags=[opts.wandb_tag], project=opts.wandb_project)
        )

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=None,
        callbacks=callbacks,
        debug=opts.debug,
    )

    _, test_interaction = trainer.eval(test_loader)
    log_stats(test_interaction, "TEST SET")
    if opts.distributed_context.is_leader:
        bart_tokenizer = BartTokenizer.from_pretrained(opts.bart_model)

        decoded_captions = bart_tokenizer.batch_decode(
            test_interaction.sender_input,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        decoded_messages = bart_tokenizer.batch_decode(
            test_interaction.message,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        test_interaction.aux_input["decoded_captions"] = decoded_captions
        test_interaction.aux_input["decoded_messages"] = decoded_messages

        dump_interaction(test_interaction, opts)

    if opts.wandb and opts.distributed_context.is_leader:
        wandb.log({"test_acc": test_interaction.aux["acc"].mean().item()}, commit=True)

    end = time.time()
    print(f"| Run took {end - start:.2f} seconds")
    print("| FINISHED JOB")


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    torch.set_deterministic(True)
    import sys

    main(sys.argv[1:])
