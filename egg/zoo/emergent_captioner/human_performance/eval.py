# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import time

import torch

import egg.core as core
from egg.core.interaction import LoggingStrategy
from egg.zoo.emergent_captioner.dataloaders import (
    CocoWrapper,
    # ConceptualCaptionsWrapper,
    FlickrWrapper,
    NoCapsWrapper,
    ImageCodeWrapper,
)
from egg.zoo.emergent_captioner.human_performance.modules import (
    ZeroShotCaptionGame,
    HumanCaptionSender,
)
from egg.zoo.emergent_captioner.human_performance.receiver import ClipReceiver
from egg.zoo.emergent_captioner.utils import (
    get_sha,
    log_stats,
    store_job_and_task_id,
)


def imagecode_loss(
    _sender_input,
    _message,
    _receiver_input,
    receiver_output,
    labels,
    _aux_input,
):
    loss = torch.zeros(1).to(receiver_output.device)
    acc = (receiver_output.argmax(dim=1) == labels).detach().float()
    return loss, {"acc": acc}


def loss(
    _sender_input,
    _message,
    _receiver_input,
    receiver_output,
    _labels,
    _aux_input,
):
    batch_size = receiver_output.shape[0]
    labels = torch.arange(batch_size, device=receiver_output.device)

    acc = (receiver_output.argmax(dim=1) == labels).detach().float()
    return torch.zeros(1), {"acc": acc}


def get_opts(params):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Run the game with pdb enabled and on only 10 batches",
    )
    parser.add_argument(
        "--recv_clip_model",
        choices=["ViT-B/16", "ViT-B/32"],
        default="ViT-B/32",
    )
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--num_workers", type=int, default=8)

    opts = core.init(arg_parser=parser, params=params)
    return opts


def main(params):
    start = time.time()

    opts = get_opts(params)
    store_job_and_task_id(opts)
    print(opts)
    print(get_sha())

    if not opts.distributed_context.is_distributed and opts.debug:
        breakpoint()

    sender = HumanCaptionSender()
    receiver = ClipReceiver(clip_model=opts.recv_clip_model)
    logging_strategy = LoggingStrategy(False, False, True, True, True, True, False)

    game = ZeroShotCaptionGame(
        sender, receiver, loss, logging_strategy=logging_strategy
    )

    trainer = core.Trainer(
        game=game,
        optimizer=torch.optim.Adam(game.parameters(), lr=opts.lr),
        train_data=None,
        debug=opts.debug,
    )

    wrappers = {
        # "conceptual": ConceptualCaptionsWrapper,
        "coco": CocoWrapper,
        "flickr": FlickrWrapper,
    }

    data_kwargs = dict(
        batch_size=opts.batch_size,
        image_size=opts.image_size,
        num_workers=opts.num_workers,
        seed=opts.random_seed,
    )
    for dataset, wrapper in wrappers.items():
        print(f"| Evaluating dataset {dataset}")
        w = wrapper()
        test_loader = w.get_split(split="test", **data_kwargs)
        _, interaction = trainer.eval(test_loader)
        log_stats(interaction, f"{dataset.upper()} TEST SET")

    nocaps_wrapper = NoCapsWrapper()
    for split in ["in-domain", "near-domain", "out-domain"]:
        test_loader = nocaps_wrapper.get_split(split=split, **data_kwargs)
        _, interaction = trainer.eval(test_loader)
        log_stats(interaction, f"NOCAPS {split} TEST SET")

    # IMAGECODE evaluation
    n_game = ZeroShotCaptionGame(
        sender, receiver, imagecode_loss, logging_strategy=logging_strategy
    )
    n_trainer = core.Trainer(
        game=n_game,
        optimizer=torch.optim.Adam(game.parameters(), lr=opts.lr),
        train_data=None,
        debug=opts.debug,
    )
    ic_wrapper = ImageCodeWrapper()
    imagecode_loader = ic_wrapper.get_split(split="test", **data_kwargs)
    _, interaction = n_trainer.eval(imagecode_loader)
    log_stats(interaction, "IMAGECODE TEST SET")

    end = time.time()
    print(f"| Run took {end - start:.2f} seconds")
    print("| FINISHED JOB")


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    import sys

    main(sys.argv[1:])
