# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import json
import random
from pathlib import Path
from typing import Callable
from PIL import Image

import torch
import torchvision
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from egg.zoo.referential_language.data_utils.collaters import (
    ContextualDistractorsCollater,
    RandomDistractorsCollater,
)


def pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class VisualGenomeDataset(torchvision.datasets.VisionDataset):
    def __init__(
        self,
        image_dir: str,
        metadata_dir: str,
        classes_path: str = "/private/home/rdessi/EGG/egg/zoo/referential_language/data_utils/classes_1600.txt",
        split: str = "train",
        transform: Callable = None,
    ):
        super(VisualGenomeDataset, self).__init__(root=image_dir, transform=transform)
        path_images = Path(image_dir)
        assert split in ["train", "val"], f"Unknown dataset split: {split}"
        path_objects = Path(metadata_dir) / f"{split}_objects.json"
        path_image_data = Path(metadata_dir) / f"{split}_image_data.json"

        with open(path_image_data) as img_in, open(path_objects) as obj_in:
            img_data = json.load(img_in)
            obj_data = json.load(obj_in)

        classes, idx = [], 0
        self.class2id, self.id2class = {}, {}
        with open(classes_path) as fin:
            for line in fin:
                syn = line.strip().split(",")
                classes.append(syn[0])
                self.id2class[idx] = syn[0]
                self.class2id[syn[0]] = idx
                idx += 1

        classes = set(classes)  # making it to a set for fast lookup
        self.samples = []
        for img, objs in zip(img_data, obj_data):
            # transforming url to local path
            # url is of this form: https://cs.stanford.edu/people/rak248/VG_100K_2/1.jpg
            # and will become {path_to_vg_folder}VG_100K_2/1.jpg
            img_path = path_images / "/".join(img["url"].split("/")[-2:])
            img_id = img["image_id"]
            h, w = img["height"], img["width"]

            obj_info = []
            for obj in objs["objects"]:
                for idx, name in enumerate(obj["names"]):
                    if name in classes:
                        obj_info.append(
                            (
                                obj["x"],
                                obj["y"],
                                obj["h"],
                                obj["w"],
                                obj["object_id"],
                                self.class2id[name],
                            )
                        )
                        break
            # NOTE: 47703 obj have > 1 string in obj["names"] taking the first for now
            # if we check and tak only the first name we have 86461 imgs and 2578322 objs
            # 2570528 if we only check and pick the first name for each object
            self.samples.append((img_path, img_id, h, w, obj_info))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, img_id, img_h, img_w, obj_info = self.samples[index]

        image = pil_loader(img_path)
        if self.transform:
            image = self.transform(image)

        boxes, labels, obj_ids = [], [], []
        for x, y, h, w, obj_id, class_id in obj_info:
            if w == 1 or h == 1:
                continue
            if (x + w) * (y + h) / (img_w * img_h) > 0.01:
                boxes.append(torch.IntTensor([x, y, h, w]))
                labels.append(torch.Tensor([class_id]))
                obj_ids.append(torch.Tensor([obj_id]))

        if len(boxes) <= 1:
            return self.__getitem__(random.randint(0, len(self) - 1))
        else:
            return (
                image,
                torch.stack(labels),
                {
                    "bboxes": torch.stack(boxes),
                    "n_objs": torch.Tensor([len(boxes)]),
                    "obj_ids": torch.stack(obj_ids),
                    "img_id": torch.IntTensor([img_id]),
                },
            )


def get_dataloader(
    image_dir: str = "/private/home/rdessi/visual_genome",
    metadata_dir: str = "/datasets01/VisualGenome1.2/061517/",
    split: str = "train",
    batch_size: int = 32,
    image_size: int = 32,
    max_objects: int = 20,
    contextual_distractors: bool = False,
    use_augmentations: bool = False,
    is_distributed: bool = False,
    seed: int = 111,
):
    to_tensor_fn = transforms.ToTensor()
    dataset = VisualGenomeDataset(
        image_dir=image_dir,
        metadata_dir=metadata_dir,
        split=split,
        transform=to_tensor_fn,
    )

    augms = None
    if use_augmentations:
        color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        transformations = [
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ]
        augms = transforms.Compose(transformations)

    if contextual_distractors:
        collater = ContextualDistractorsCollater(max_objects, image_size, augms)
    else:
        collater = RandomDistractorsCollater(max_objects, image_size, dataset, augms)

    sampler = None
    if is_distributed:
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=True, seed=seed)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        collate_fn=collater,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    return loader
