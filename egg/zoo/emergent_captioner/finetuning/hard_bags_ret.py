import clip
import pickle


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
    wrapper = name2wrapper[opts.train_dataset](captions_type = config["captions_type"], dataset_dir = opts.dataset_dir, jatayu = opts.jatayu, neg_mining = config["neg_mining"])
    
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
    idx2cocoid = {v : k for k, v in cocoid2idx.items()}
    # GT image and caption CLIP feats
    img_feats = torch.load(os.path.join(data_dir, "img_feats", f"coco_test_vitl14.pt"))
    text_feats = torch.load(os.path.join(data_dir, "text_feats", f"{config['captions_type']}_test_vitl14.pt"))
    # sender_input, aux : cocoid, captions
    recall_1 = []
    recall_5 = []
    clip_s = []
    mean_rank = []
    median_rank = []
    loss = DiscriminativeLoss()

#------------------------------------------------------------------------------------------------------------------------------------------------
    """RANDOM 100 batch_size batch using GT"""

    # for batch in tqdm(test_loader, total = len(test_loader)):
    #     _,_,_, aux = batch
    #     clip_idx = [cocoid2idx[str(cocoid.item())] for cocoid in aux["cocoid"]]

    #     batch_img_feats = img_feats[clip_idx]
    #     batch_text_feats = text_feats[clip_idx]
    #     batch_img_feats = batch_img_feats / batch_img_feats.norm(dim=-1, keepdim = True)
    #     batch_text_feats = batch_text_feats / batch_text_feats.norm(dim=-1, keepdim = True)

    #     for i in range(batch_text_feats.shape[1]):
    #         _, acc = loss(batch_text_feats[:, i, :],batch_img_feats, False, True, None)
    #         recall_1.append(acc['acc'].mean().item())
    #         recall_5.append(acc['acc_5'].mean().item())
    #         clip_s.append(acc['clip_s'].mean().item())

#------------------------------------------------------------------------------------------------------------------------------------------------
    ## """RANDOM 100 batch_size batch using MODEL PREDS"""
    
    # #CLIP 
    # model_name = "ViT-L/14@336px"
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model, preprocess = clip.load(model_name, device=device)
    # model.eval()

    # #get preds
    # captioner = "coco_cider_lora_rank32_g_baseline"

    # bag_dir = "/home/manugaur/nips_benchmark/bags/clip_vitl14_mm_coco"
    # preds_path = f"/home/manugaur/EGG/inference_preds/{captioner}.pkl"

    # with open(preds_path, "rb") as f:
    #     preds = pickle.load(f)
    # get_acc_5 = True


    # for batch in tqdm(test_loader, total = len(test_loader)):
    #     _,_,_, aux = batch
    #     clip_idx = [cocoid2idx[str(cocoid.item())] for cocoid in aux["cocoid"]]

    #     batch_img_feats = img_feats[clip_idx]
    #     batch_caps = [preds[int(idx2cocoid[i])] for i in clip_idx]
    #     with torch.no_grad():
    #         batch_text_feats = model.encode_text(clip.tokenize(batch_caps, context_length=77, truncate=True).to(device))

        
    #     _, acc = loss(batch_text_feats ,batch_img_feats, False, get_acc_5,  None)
    #     recall_1.append(acc['acc'].mean().item())
    #     recall_5.append(acc['acc_5'].mean().item())
    #     clip_s.append(acc['clip_s'].mean().item())
    #     mean_rank.append(acc['mean_rank'].item())
    #     median_rank.append(acc['median_rank'].item())

    
    #     del batch_img_feats
    #     del batch_text_feats
    #     torch.cuda.empty_cache()
    # print("\n")
    # print(f"{captioner}:")

#------------------------------------------------------------------------------------------------------------------------------------------------
    """RETRIEVAL WITHIN HARD BAGS using GT"""
    # USE_GREEDY_CAP = True # use greedy GT not the sampled ones.
    # RECALL_PER_BAG = True
    # bag_dir = "/home/manugaur/nips_benchmark/bags/clip_vitl14_mm_coco"
    # bag_size = 3
    # threshold = 0
    # captioner = f"{config['captions_type']}_gt"

    # with open(os.path.join(bag_dir, f"bsz_{bag_size}_thresh_{threshold}.json"), "r") as f:
    #     listofbags = json.load(f)
    # print(len(listofbags))

    # benchmark = []
    # for bag in listofbags:
    #     benchmark.append([cocoid2idx[str(cocoid)] for cocoid in bag])
   
    # for idx , bag in tqdm(enumerate(benchmark), total = len(benchmark)):
    #     if not RECALL_PER_BAG and idx == num_bags:
    #         break
        
    #     bag_img_feats = img_feats[bag]
    #     bag_text_feats = text_feats[bag]
    #     bag_img_feats = bag_img_feats / bag_img_feats.norm(dim=-1, keepdim = True)
    #     bag_text_feats = bag_text_feats / bag_text_feats.norm(dim=-1, keepdim = True)
        
    #     if USE_GREEDY_CAP:
    #         bag_text_feats = bag_text_feats[:, 0, :]
    #         _, acc = loss(bag_text_feats,bag_img_feats, False, False, None)
    #         if RECALL_PER_BAG:
    #             recall_1.append([_.item() for _ in acc['acc']])
    #         else:
    #             recall_1.append(acc['acc'].mean().item())

    #     else: 
    #         for i in range(bag_text_feats.shape[1]):
    #             _, acc = loss(bag_text_feats[:, i, :],bag_img_feats, False, False, None)
                
    #             if RECALL_PER_BAG:
    #                 recall_1.append([_.item() for _ in acc['acc']])
    #             else:
    #                 recall_1.append(acc['acc'].mean().item())
    #             # recall_5.append(acc['acc_5'].mean().item())
    #             clip_s.append(acc['clip_s'].mean().item())

#------------------------------------------------------------------------------------------------------------------------------------------------
    """RETRIEVAL WITHIN HARD BAGS using MODEL PREDS"""
    
    # 3,1,300 | 5,1,122 | 7,2,125 | 10,3,81| 20,7,10
    RECALL_PER_BAG = True
    bag_size, threshold, num_bags = 3, 0, 30
    captioner =  "blip2mistral_sr_baseline"#"coco_sr_bsz_200"
 
    #CLIP 
    model_name = "ViT-L/14@336px"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device)
    model.eval()

    bag_dir = "/home/manugaur/nips_benchmark/bags/clip_vitl14_mm_coco"
    preds_path = f"/home/manugaur/EGG/inference_preds/{captioner}.pkl"

    with open(preds_path, "rb") as f:
        preds = pickle.load(f)
    
    with open(os.path.join(bag_dir, f"bsz_{bag_size}_thresh_{threshold}.json"), "r") as f:
        listofbags = json.load(f)

    get_acc_5 = bag_size > 5


    benchmark = []
    for bag_idx, bag in enumerate(listofbags):
        if bag_idx ==num_bags:
            break
        benchmark.append([cocoid2idx[str(cocoid)] for cocoid in bag])
   
    for bag_idx , bag in tqdm(enumerate(benchmark), total =num_bags):

        bag_img_feats = img_feats[bag]
        bag_img_feats = bag_img_feats / bag_img_feats.norm(dim=-1, keepdim = True)
    #     batch_text_feats = batch_text_feats / batch_text_feats.norm(dim=-1, keepdim = True)
        bag_caps =  [preds[int(idx2cocoid[clip_idx])] for clip_idx in bag]
        with torch.no_grad():
            bag_text_feats = model.encode_text(clip.tokenize(bag_caps, context_length=77, truncate=True).to(device))
            bag_text_feats = bag_text_feats / bag_text_feats.norm(dim=-1, keepdim=True)
        
        _, acc = loss(bag_text_feats ,bag_img_feats,  False, get_acc_5, None)
        
        if RECALL_PER_BAG:
            recall_1.append([_.item() for _ in acc['acc']])
        else:
            recall_1.append(acc['acc'].mean().item())
        if get_acc_5:
            recall_5.append(acc['acc_5'].mean().item())
        clip_s.append(acc['clip_s'].mean().item())
        mean_rank.append(acc['mean_rank'].item())
        median_rank.append(acc['median_rank'].item())

        del bag_img_feats
        del bag_text_feats
        torch.cuda.empty_cache()
    print("\n")
    print(f"bag size : {bag_size}")
    print(f"threshold : {threshold}")
    print(f"num bags : {num_bags}")
    print("\n")
#------------------------------------------------------------------------------------------------------------------------------------------------
    if RECALL_PER_BAG:
        with open(f"/home/manugaur/nips_benchmark/recall_per_bag/bsz_{bag_size}_thresh_{threshold}_{captioner}.json", "w") as f:
            json.dump(recall_1, f)
    # print(f"{round(np.array(recall_1).mean()*100,2)}/ {np.array(mean_rank).mean():.2f}/ {np.array(median_rank).mean():.2f}")
    print(f"Recall@1 : {round(np.array(recall_1).mean()*100,2)}")
    # print(f"Recall@5 : {round(np.array(recall_5).mean()*100,2)}")
    # print(f"Mean rank : {np.array(mean_rank).mean():.2f}")
    # print(f"Median rank : {np.array(median_rank).mean():.2f}")
    print(f"CLIP score : {round(np.array(clip_s).mean(), 2):.2f}")
    

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
    config["captions_type"] = "coco"
    config["opts"]["batch_size"]= 200
    print(f"Self Retrieval using {config['captions_type']} captions")
    main(params, config)