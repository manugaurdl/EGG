import sys
import time
import os
import torch
import numpy as np
import random
from transformers import get_linear_schedule_with_warmup
from egg.zoo.emergent_captioner.finetuning.utils import get_config, set_data_dir, get_cl_args, init_wandb
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

seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

# "MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK", "LOCAL_RANK"
# os.environ["CUDA_VISIBLE_DEVICES"] = str((0))


def main(params, config):
    start = time.time()
    opts = get_common_opts(params=params)
    opts.jatayu = os.path.isdir("/home/manugaur")
    opts.loss_type= config['train_method']
    store_job_and_task_id(opts)
    setup_for_distributed(opts.distributed_context.is_leader)

    if  opts.distributed_context.local_rank ==0:
        init_wandb(config)
    
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
    wrapper = name2wrapper[opts.train_dataset](config["captions_type"], opts.dataset_dir, opts.jatayu)
    data_kwargs = dict(
        batch_size=opts.batch_size,
        transform=get_transform(opts.sender_image_size, opts.recv_image_size),
        num_workers=0,
        seed=opts.random_seed,
        debug = config['DEBUG'],
        mle_train = config["train_method"] =="mle",
        max_len_token = opts.max_len,
        prefix_len = config["prefix_len"],
    )
    train_loader = wrapper.get_split(split="train", **data_kwargs)
    val_loader = wrapper.get_split(split="val",**data_kwargs)
    test_loader = wrapper.get_split(split="test", **data_kwargs)
    # train_loader = wrapper.get_split(split="train",shuffle = not config['DEBUG'], debug = config['DEBUG'], **data_kwargs)
    # val_loader = wrapper.get_split(split="val",shuffle = not config['DEBUG'], debug = config['DEBUG'],  **data_kwargs)

    game = build_game(opts, config)
    # print_grad_info(game)
    
    optimizer = torch.optim.AdamW(game.sender.parameters(), lr=opts.lr)


    if config["train_method"] == "mle":
        total_steps = opts.n_epochs* len(train_loader)
        scheduler = get_linear_schedule_with_warmup(
                    optimizer, num_warmup_steps=int(total_steps * config["warmup_ratio"]), num_training_steps= total_steps)

        trainer = core.Trainer(
            game=game,
            optimizer=optimizer,
            train_data=train_loader,
            optimizer_scheduler = scheduler,
            validation_data =val_loader,
            callbacks=[
                ConsoleLogger(as_json=True, print_train_loss=True),
                ModelSaver(opts),
            ],
            debug=opts.debug,
        )
    else:
        trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data =val_loader,
        callbacks=[
            ConsoleLogger(as_json=True, print_train_loss=True),
            ModelSaver(opts),
        ],
        debug=opts.debug,
        )  
    
    if opts.distributed_context.is_distributed:
        trainer.game = trainer.game.module

    if opts.captioner_model == "clipcap":   
        trainer.game.sender.patch_model(batch_size = opts.batch_size, prefix_len = config['prefix_len'], )

    trainer.train(config, opts)

    # _, test_interaction, test_reward = trainer.eval(val_loader)

    # log_stats(test_interaction, "TEST SET")
    # dump_interaction(test_interaction, opts, name="finetuned_")

    end = time.time()
    print(f"| Run took {end - start:.2f} seconds")
    print("| FINISHED JOB")


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    # torch.set_deterministic(True)    
    if "LOCAL_RANK" in os.environ:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    
    else:
        config_filename = f"egg/zoo/emergent_captioner/finetuning/configs/{sys.argv[1:][0]}.yml"    # get this from sys args 
    
    config = get_config(config_filename)
    config = set_data_dir(config)
    params = get_cl_args(config)


    main(params, config)