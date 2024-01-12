# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Callable

import torch
import torch.distributed as dist
from PIL import Image

from egg.zoo.emergent_captioner.dataloaders.utils import MyDistributedSampler


class CocoDataset:
    def __init__(self, root, samples, transform):
        self.root = root
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)
        # return 1024

    def __getitem__(self, idx):

        file_path, captions, image_id = self.samples[idx]
        # try:
        # print("*********************************")
        # print(self.root)
        # print(str(file_path))
        image = Image.open(str(file_path)).convert("RGB")
        # image = Image.open(os.path.join(self.root, file_path)).convert("RGB")
        # except:
            # import ipdb;ipdb.set_trace()

        sender_input, recv_input = self.transform(image)

        aux = {"img_id": torch.tensor([image_id]), "captions": captions[:5]}

        return sender_input, torch.tensor([idx]), recv_input, aux


class CocoWrapper:
    def __init__(self, dataset_dir: str = None, jatayu: bool = False):
        if dataset_dir is None:
            dataset_dir = "/checkpoint/rdessi/datasets/coco"
        self.dataset_dir = Path(dataset_dir)

        self.split2samples = self._load_splits(jatayu)

    def _load_splits(self, jatayu):
        dataset_coco_path = "/ssd_scratch/cvit/manu/img_cap_self_retrieval_clip/annotations/dataset_coco.json"
        if jatayu:
            dataset_coco_path = "/home/manugaur/img_cap_self_retrieval/data/annotations/dataset_coco.json"
        with open(dataset_coco_path) as f:
            annotations = json.load(f)
        split2samples = defaultdict(list)
        for img_ann in annotations["images"]:
            # breakpoint()
            file_path = self.dataset_dir / img_ann["filepath"] / img_ann["filename"]
            captions = [x["raw"] for x in img_ann["sentences"]]
            img_id = img_ann["imgid"]
            split = img_ann["split"]

            split2samples[split].append((file_path, captions, img_id))
        if "restval" in split2samples:
            split2samples["train"] += split2samples["restval"]

        for k, v in split2samples.items():
            print(f"| Split {k} has {len(v)} elements.")
        return split2samples

    def get_split(
        self,
        split: str,
        batch_size: int,
        transform: Callable,
        num_workers: int = 8,
        shuffle: bool = None,
        seed: int = 111,
    ):

        samples = self.split2samples[split]
        assert samples, f"Wrong split {split}"

        ds = CocoDataset(self.dataset_dir, samples, transform=transform)

        sampler = None
        if dist.is_initialized():
            if shuffle is None:
                shuffle = split != "test"
            sampler = MyDistributedSampler(
                ds, shuffle=shuffle, drop_last=True, seed=seed
            )

        if shuffle is None:
            shuffle = split != "test" and sampler is None

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
