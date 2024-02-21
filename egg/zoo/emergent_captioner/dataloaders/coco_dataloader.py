# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Callable
import pickle
import time
import torch
import json
import torch.distributed as dist
from PIL import Image
from transformers import GPT2Tokenizer
from tqdm import tqdm
from egg.zoo.emergent_captioner.dataloaders.utils import MyDistributedSampler
from torch.utils.data.distributed import DistributedSampler


def open_pickle(path: str):
    with open(path, "rb") as f:
        file = pickle.load(f)
    return file 

class CocoDataset:
    def __init__(self, root, samples, mle_train, split,captions_type, max_len_token, prefix_len, transform, debug):
        self.root = root
        self.samples = samples
        self.transform = transform
        self.debug = debug
        self.split = split
        self.mle_train = mle_train
        self.max_len_token = max_len_token
        self.prefix_len = prefix_len
        self.captions_type = captions_type
        if self.mle_train:
            self.path2tokens = os.path.join(root, f"tokenized_caps/{self.captions_type}/{self.split}")
            # self.id2tokens = torch.load
            pass
    def __len__(self):
        if self.debug:
            return 60
        else:
            return len(self.samples)
    
    def pad(self,tokens):
                     
        padding = self.max_len_token - tokens.shape[-1]

        if padding>0:
            pad = torch.zeros(padding)
            pad = pad.masked_fill(pad ==0, -1) 
            tokens = torch.cat((tokens, pad)).int() # tokens is padded with -1.

            # ### padded tokens replace the tokens. Here the padding is done by -1. But the tokens returned by the method have padding with 0.
            # if not self.lazy_load:
            #     self.tokenized_captions[idx][cap_idx] = tokens
        else:
            # if caption > max_len, truncate it 
            tokens = tokens[:self.max_len_token]
            # if not self.lazy_load:
            #     self.tokenized_captions[idx][cap_idx] = tokens
            
        mask = tokens.ge(0) #True for indices > 0 i,e padded indices = False
        tokens[~mask] =0  # padding now done with 0
        mask = torch.cat((torch.ones(self.prefix_len),mask)) 
        
        return (tokens, mask)


    def get_tokens(self,cocoid):
        path =  f"{self.path2tokens}/{cocoid}"
        # compare with calling self.pad 2 times with list comprehension and stack
        # compare with storing all 5 captions as single tensor, with padding as -1
        tokens = []
        masks = []
        for i in range(4):
            t, m = self.pad(torch.load(path + f"_{i}.pt"))
            tokens.append(t)
            masks.append(m)

        return torch.stack(tokens), torch.stack(masks)

    def __getitem__(self, idx):
        file_path, captions, image_id = self.samples[idx]
        # image = Image.open(str(file_path)).convert("RGB")

        image = Image.open(os.path.join(self.root, file_path)).convert("RGB")
        sender_input, recv_input = self.transform(image)
        if self.mle_train:
            padded_tokens, mask = self.get_tokens(image_id)
            aux = {"cocoid": torch.tensor([image_id]), "captions": captions[:5], "tokens": padded_tokens, "mask" : mask}
        else:
            aux = {"cocoid": torch.tensor([image_id]), "captions": captions[:5]}

        return sender_input, torch.tensor([idx]), recv_input, aux


class CocoWrapper:

    def __init__(self, captions_type : bool, dataset_dir: str = None, jatayu: bool = False):
        self.num_omitted_ids = 0
        if dataset_dir is None:
            dataset_dir = "/checkpoint/rdessi/datasets/coco"
        self.dataset_dir = Path(dataset_dir)
        self.captions_type = captions_type
        if self.captions_type != "coco":
            self.id2caption = open_pickle(os.path.join(dataset_dir, f"synthetic_data/cocoid2caption_{self.captions_type}_preproc.pkl"))
            assert isinstance(list(self.id2caption.values())[0], list), "cocoid2cap is not id --> list of caps"
        self.split2samples = self._load_splits(jatayu) # {test,val,train,restval} --> {test[0] :(img_path, list of 5 caps, cocoid) }
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        print(f"{self.num_omitted_ids} cocoids are removed during preproc for {self.captions_type} captions")

    def tokenize(self,split):
        """
        self.split2samples[split] : list of [img_path, list_of_caps, cocoid]
        """

        self.all_len = []
        save_dir = os.path.join(self.dataset_dir, f"tokenized_caps/{self.captions_type}/{split}/")
        if not os.path.isdir(save_dir) or len(os.listdir(save_dir))<1000:
            print(f"tokenizing {split} captions...")
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)    
            for instance in tqdm(self.split2samples[split], total = len(self.split2samples[split])):
                cocoid = instance[2]
                captions = instance[1]
                if isinstance(captions, tuple) or isinstance(captions, list):
                    for idx, cap in enumerate(captions):
                        token = torch.tensor(self.tokenizer.encode(cap),dtype=torch.int)
                        torch.save(token, os.path.join(save_dir, f"{cocoid}_{idx}.pt"))
                    # tokens = [torch.tensor(self.tokenizer.encode(cap),dtype=torch.int) for cap in caption]
                        self.all_len.append(token.shape[-1])
            with open(os.path.join(self.dataset_dir, f"tokenized_caps/{self.captions_type}/{split}/all_len.json"), "w") as f:
                json.dump(self.all_len, f)

        else:
            print(print(f"tokenized {split} captions exist"))

    def _load_splits(self, jatayu):
        if jatayu:
            path2ann = "/home/manugaur/img_cap_self_retrieval/data/annotations/dataset_coco.json"
        else:
            path2ann = "/ssd_scratch/cvit/manu/img_cap_self_retrieval_clip/annotations/dataset_coco.json"

        with open(path2ann) as f:
            annotations = json.load(f)
        split2samples = defaultdict(list)
        
        for img_ann in annotations["images"]:
            # import ipdb;ipdb.set_trace()
            file_path = self.dataset_dir / img_ann["filepath"] / img_ann["filename"]
            cocoid = img_ann["cocoid"]
            try:
                if self.captions_type =="coco":
                    captions = [x["raw"] for x in img_ann["sentences"]]
                else:
                    captions = self.id2caption[cocoid]
            except KeyError:
                self.num_omitted_ids+=1
            # img_id = img_ann["imgid"]
            split = img_ann["split"]

            split2samples[split].append((file_path, captions, cocoid))
        if "restval" in split2samples:
            split2samples["train"] += split2samples["restval"]

        for k, v in split2samples.items():
            print(f"| Split {k} has {len(v)} elements.")
        return split2samples

    def get_split(
        self,
        split: str,
        debug : bool,
        batch_size: int,
        mle_train : bool,
        max_len_token : int,
        prefix_len : int,
        is_dist_leader : bool,
        transform: Callable,
        num_workers: int = 8,
        seed: int = 111,
    ):
        if mle_train:
            self.tokenize(split)
        shuffle = not debug
        samples = self.split2samples[split]
        assert samples, f"Wrong split {split}"

        ds = CocoDataset(self.dataset_dir, samples, mle_train, split, self.captions_type, max_len_token, prefix_len,transform=transform, debug = debug)

        sampler = None

        if dist.is_initialized() and split =="train":
            print(f"{split} data is distributed.")
            if shuffle is None:
                shuffle = split != "test"

            # sampler = MyDistributedSampler(
            #     ds, shuffle=shuffle, drop_last=True, seed=seed
            # )
            sampler = DistributedSampler(ds, num_replicas=int(os.environ["LOCAL_WORLD_SIZE"]), rank= int(os.environ["LOCAL_RANK"]), shuffle=False, drop_last=False)


        if shuffle is None:
            shuffle = split != "test" and sampler is None

        if sampler is not None :
            shuffle=None
        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

        return loader


if __name__ == "__main__":
    wrapper = CocoWrapper()
    dl = wrapper.get_split(
        split="test",
        batch_size=10,
        image_size=224,
        shuffle=False,
        num_workers=8,
    )

    for i, elem in enumerate(dl):
        breakpoint()
