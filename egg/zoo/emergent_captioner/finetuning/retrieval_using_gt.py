import sys
import time
import os
import json 
import torch
import numpy as np
from tqdm import tqdm
import random
from transformers import get_linear_schedule_with_warmup
from egg.zoo.emergent_captioner.finetuning.utils import get_config, process_config, get_cl_args, init_wandb, get_best_state_dict, int2mil
from egg.zoo.emergent_captioner.finetuning.losses import get_loss, DiscriminativeLoss

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
    # opts.loss_type= config['train_method']

    print(get_sha())
    
    name2wrapper = {
        "conceptual": ConceptualCaptionsWrapper,
        "coco": CocoWrapper,
        "flickr": FlickrWrapper,
    }
    # args
    config["neg_mining"]["do"] = False
    wrapper = name2wrapper[opts.train_dataset](captions_type = config["captions_type"], dataset_dir = opts.dataset_dir, jatayu = opts.jatayu, neg_mining = config["neg_mining"], ONLY_VAL= False)
    
    data_kwargs = dict(
        batch_size=config["opts"]["batch_size"],
        transform=get_transform(opts.sender_image_size, opts.recv_image_size),
        num_workers=config["num_workers"],
        seed=opts.random_seed,
        debug = False,
        mle_train = config["train_method"] =="mle",
        max_len_token = opts.max_len,
        prefix_len = config["prefix_len"],
        is_dist_leader = opts.distributed_context.is_leader,
    )

    test_loader = wrapper.get_split(split="test", caps_per_img = 5, neg_mining = False, **data_kwargs)
    
    data_dir = "/home/manugaur/nips_benchmark/"
    

    with open(os.path.join(data_dir, "misc_data", f"{config['captions_type']}_test_cocoid2idx.json"), "r") as f:
        cocoid2idx = json.load(f)

    img_feats = torch.load(os.path.join(data_dir, "img_feats", f"coco_test_vitl14.pt"))
    text_feats = torch.load(os.path.join(data_dir, "text_feats", f"{config['captions_type']}_test_vitl14.pt"))
    # sender_input, aux : cocoid, captions
    recall_1 = []
    recall_5 = []
    clip_s = []
    loss = DiscriminativeLoss()

    # RANDOM 100 batch_size batch.

    # for batch in tqdm(test_loader, total = len(test_loader)):
    #     _,_,_, aux = batch
    #     clip_idx = [cocoid2idx[str(cocoid.item())] for cocoid in aux["cocoid"]]

    #     batch_img_feats = img_feats[clip_idx]
    #     batch_text_feats = text_feats[clip_idx]
        
    #     for i in range(batch_text_feats.shape[1]):
    #         _, acc = loss(batch_text_feats[:, i, :],batch_img_feats, None)
    #         recall_1.append(acc['acc'].mean().item())
    #         recall_5.append(acc['acc_5'].mean().item())
    #         clip_s.append(acc['clip_s'].mean().item())


    # RETRIEVAL WITHIN HARD BAGS

    bag_dir = "/home/manugaur/nips_benchmark/bags/clip_vitl14_mm_coco"
    bag_size = 7
    threshold = 3
    
    with open(os.path.join(bag_dir, f"bsz_{bag_size}_thresh_{threshold}.json"), "r") as f:
        listofbags = json.load(f)

    benchmark = []
    for bag in listofbags:
        benchmark.append([cocoid2idx[str(cocoid)] for cocoid in bag])
   
    for idx , bag in tqdm(enumerate(benchmark), total =128):
        if idx == 128:
            break
        bag_img_feats = img_feats[bag]
        bag_text_feats = text_feats[bag]
        
        for i in range(bag_text_feats.shape[1]):
            _, acc = loss(bag_text_feats[:, i, :],bag_img_feats, None)
            
            recall_1.append(acc['acc'].mean().item())
            # recall_5.append(acc['acc_5'].mean().item())
            # clip_s.append(acc['clip_s'].mean().item())

    
    print(f"Recall@1 : {np.array(recall_1).mean()}")
    # print(f"Recall@5 : {np.array(recall_5).mean()}")
    # print(f"CLIP score : {np.array(clip_s).mean()}")
        # batch_text_feats.view(batch_text_feats.shape[1], batch_text_feats.shape[0] , -1)[0]

    
    end = time.time()
    print(f"| Run took {end - start:.2f} seconds")
    print("| FINISHED JOB")


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    use_ddp = False    

    if "LOCAL_RANK" in os.environ:
        use_ddp = True
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    config_filename = f"egg/zoo/emergent_captioner/finetuning/configs/{sys.argv[1:][0]}.yml"
    config = get_config(config_filename)
    config = process_config(config, use_ddp, sys.argv[1:])
    params = get_cl_args(config)
    config["captions_type"] = "blip2mistral"
    config["opts"]["batch_size"]= 100
    print(f"Self Retrieval using {config['captions_type']} captions")
    main(params, config)