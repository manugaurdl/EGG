# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

WANDB = True
WANDB_NAME = "cider_optim_b_greedy_t_1e-1_my_sample_maxlen_20_lr_1e-6"
DEBUG = False
INIT_VAL = True
CIDER_OPTIM = True
GREEDY_BASELINE = True

import wandb
import time
import os
import torch
import egg.core as core
from egg.core import ConsoleLogger
from egg.zoo.emergent_captioner.dataloaders import (
    CocoWrapper,
    ConceptualCaptionsWrapper,
    FlickrWrapper,
    get_transform,
)
from egg.zoo.emergent_captioner.finetuning.game import build_game
from egg.zoo.emergent_captioner.finetuning.opts import get_common_opts
from egg.zoo.emergent_captioner.finetuning.utils import ModelSaver
from egg.zoo.emergent_captioner.utils import (
    dump_interaction,
    get_sha,
    log_stats,
    print_grad_info,
    setup_for_distributed,
    store_job_and_task_id,
)


def main(params):
    print("Inside main")
    # import ipdb;ipdb.set_trace()

    start = time.time()
    opts = get_common_opts(params=params)
    if CIDER_OPTIM:
        opts.loss_type= "cider"
    store_job_and_task_id(opts)
    setup_for_distributed(opts.distributed_context.is_leader)
    # print(opts)
    print(get_sha())

    if not opts.distributed_context.is_distributed and opts.debug:
        breakpoint()

    name2wrapper = {
        "conceptual": ConceptualCaptionsWrapper,
        "coco": CocoWrapper,
        "flickr": FlickrWrapper,
    }
    # args
    jatayu  = not os.path.isdir("/ssd_scratch/cvit")
    wrapper = name2wrapper[opts.train_dataset](opts.dataset_dir, jatayu)

    data_kwargs = dict(
        batch_size=opts.batch_size,
        transform=get_transform(opts.sender_image_size, opts.recv_image_size),
        num_workers=opts.num_workers,
        seed=opts.random_seed,
    )
    train_loader = wrapper.get_split(split="train", debug = DEBUG, **data_kwargs)
    val_loader = wrapper.get_split(split="val",debug = DEBUG,  **data_kwargs)

    game = build_game(opts)
    # print_grad_info(game)

    optimizer = torch.optim.Adam(game.sender.parameters(), lr=opts.lr)
    # import ipdb;ipdb.set_trace()
    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data  =val_loader,
        callbacks=[
            ConsoleLogger(as_json=True, print_train_loss=True),
            ModelSaver(opts),
        ],
        debug=opts.debug,
    )
    if opts.captioner_model == "clipcap":
        trainer.game.sender.patch_model()

    trainer.train(opts.n_epochs, WANDB, INIT_VAL, GREEDY_BASELINE, opts)

    # _, test_interaction, test_reward = trainer.eval(val_loader)

    # log_stats(test_interaction, "TEST SET")
    # dump_interaction(test_interaction, opts, name="finetuned_")

    end = time.time()
    print(f"| Run took {end - start:.2f} seconds")
    print("| FINISHED JOB")


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    # torch.set_deterministic(True)
    import sys
    if WANDB:
        wandb.init(entity= "manugaur", project="emergent_captioner")
        wandb.run.name = WANDB_NAME

    
    main(sys.argv[1:])