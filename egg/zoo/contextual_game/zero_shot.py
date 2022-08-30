# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time

import torch
from transformers import GPT2Tokenizer, AdamW

import egg.core as core
from egg.core import ConsoleLogger

# from egg.zoo.contextual_game.data import get_dataloader
# from egg.zoo.contextual_game.coco_dataloader import get_dataloader
from egg.zoo.contextual_game.flickr_dataloader import get_dataloader
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

    test_loader = get_dataloader(image_size=opts.image_size, split="test", batch_size=1)

    """
    image_dir = "/datasets01/COCO/060817/val2014"
    metadata_dir = "/datasets01/COCO/060817/annotations/captions_val2014.json"
    test_loader = get_dataloader(
        image_dir=image_dir,
        metadata_dir=metadata_dir,
        batch_size=1,
        image_size=opts.image_size,
    )
    test_loader = get_dataloader(
        image_dir=opts.image_dir,
        metadata_dir=opts.metadata_dir,
        batch_size=opts.batch_size,
        image_size=opts.image_size,
        split="test",
        num_workers=opts.num_workers,
        is_distributed=opts.distributed_context.is_distributed,
        seed=opts.random_seed,
    )
    """

    game = build_game(opts)

    trainer = core.Trainer(
        game=game,
        optimizer=AdamW(game.receiver.parameters(), lr=opts.lr),
        train_data=None,
        callbacks=[ConsoleLogger(as_json=True, print_train_loss=True)],
        debug=opts.debug,
    )

    _, test_interaction = trainer.eval(test_loader)
    log_stats(test_interaction, "TEST SET")

    if opts.distributed_context.is_leader:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        # remove custom padding (it's -1) since gpt2 does not have padding
        messages = [
            [token for token in msg if token >= 0]
            for msg in test_interaction.message.tolist()
        ]
        test_interaction.aux_input["decoded_messages"] = tokenizer.batch_decode(
            messages,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        captions = [
            [token for token in caption if token >= 0]
            for caption in test_interaction.aux_input["captions"].tolist()
        ]
        test_interaction.aux_input["decoded_captions"] = tokenizer.batch_decode(
            captions,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        dump_interaction(test_interaction, opts)

    end = time.time()
    print(f"| Run took {end - start:.2f} seconds")
    print("| FINISHED JOB")


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    torch.set_deterministic(True)
    import sys

    main(sys.argv[1:])
