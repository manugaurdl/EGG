# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license fod in the
# LICENSE file in the root directory of this source tree.

import json
import random
from math import ceil
from pathlib import Path
from typing import Callable
from PIL import Image

import torch
from torch.utils.data import IterableDataset
from torchvision import transforms
from torchvision.transforms.functional import crop


def pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class _ImageIterator:
    def __init__(
        self,
        samples,
        class2id,
        transform=None,
        max_objects=20,
        image_size=64,
        shuffle=False,
    ):
        self.samples = samples
        if shuffle:
            random.shuffle(self.samples)
        self.class2id = class2id
        self.transform = transform
        self.max_objects = max_objects
        self.image_size = image_size
        self.curr_idx = -1

    def __next__(self):
        self.curr_idx += 1
        if self.curr_idx >= len(self.samples):
            self.curr_idx = 0
            raise StopIteration

        img_path, obj_info, img_id = self.samples[self.curr_idx]

        image = pil_loader(img_path)
        if self.transform:
            image = self.transform(image)

        resizer = transforms.Resize(size=(self.image_size, self.image_size))
        cropped_objs, labels = [], []
        last_obj = min(self.max_objects, len(obj_info))
        for obj in obj_info[:last_obj]:
            labels.append(self.class2id[obj["names"][0]])
            y, x, h, w = obj["y"], obj["x"], obj["h"], obj["w"]
            cropped_objs.append(resizer(crop(image, y, x, h, w)))

        agent_input = torch.stack(cropped_objs)
        labels = torch.Tensor(labels)
        return agent_input, labels, torch.Tensor([1])

    def __iter__(self):
        return self


class VisualGenomeDataset(IterableDataset):
    def __init__(
        self,
        image_dir: str,
        metadata_dir: str,
        classes_path: str = "/private/home/rdessi/EGG/egg/zoo/referential_language/data_utils/classes_1600.txt",
        split: str = "train",
        transform: Callable = None,
        max_objects=20,
        image_size=64,
    ):
        super(VisualGenomeDataset, self).__init__()
        path_images = Path(image_dir)
        path_metadata = Path(metadata_dir) / f"{split}_objects.json"
        path_image_data = Path(metadata_dir) / f"{split}_image_data.json"

        with open(path_image_data) as img_in, open(path_metadata) as metadata_in:
            img_data, img_metadata = json.load(img_in), json.load(metadata_in)
        assert len(img_data) == len(img_metadata)

        get_name = lambda line: line.strip().split(",")[0]
        with open(classes_path) as fin:
            self.class2id = {get_name(line): idx for idx, line in enumerate(fin)}

        self.samples = []
        for img, objs_data in zip(img_data, img_metadata):
            assert img["image_id"] == objs_data["image_id"]
            img_path = path_images / "/".join(img["url"].split("/")[-2:])

            objs = self._filter_objs(img, objs_data["objects"])
            if len(objs) > 2:
                self.samples.append((img_path, objs, objs_data["image_id"]))

        self.transform = transform
        self.max_objects = max_objects
        self.image_size = image_size
        self.shuffle = split == "train"

    def _filter_objs(self, img, objs):
        filtered_objs = []
        for obj in objs:
            o_name = next(filter(lambda x: x in self.class2id, obj["names"]), None)
            if o_name is None:
                continue
            obj["names"] = [o_name]

            x, y, h, w = obj["x"], obj["y"], obj["h"], obj["w"]
            img_area = img["width"] * img["height"]
            obj_area = (x + w) * (y + h)
            is_big = obj_area / img_area > 0.01 and w > 1 and h > 1
            if is_big:
                filtered_objs.append(obj)
        return filtered_objs

    def __iter__(self):
        iter_start = 0
        iter_end = len(self.samples)

        worker_info = torch.utils.data.get_worker_info()
        if worker_info:  # num_workers is > 0
            per_worker = int(ceil(len(self.samples) / float(worker_info.num_workers)))
            iter_start = worker_info.id * per_worker
            iter_end = min(iter_start + per_worker, len(self.samples))

        return _ImageIterator(
            samples=self.samples[iter_start:iter_end],
            class2id=self.class2id,
            transform=self.transform,
            max_objects=self.max_objects,
            image_size=self.image_size,
            shuffle=self.shuffle,
        )

    def __len__(self):
        return len(self.samples)


def get_dataloader(
    image_dir: str = "/datasets01/VisualGenome1.2/061517/",
    metadata_dir: str = "/private/home/rdessi/visual_genome",
    split: str = "train",
    image_size: int = 32,
    max_objects: int = 20,
    # contextual_distractors: bool = False,
):
    transform = transforms.ToTensor()
    dataset = VisualGenomeDataset(
        image_dir,
        metadata_dir,
        split=split,
        transform=transform,
        max_objects=max_objects,
        image_size=image_size,
    )
    return torch.utils.data.DataLoader(dataset, num_workers=12, pin_memory=True)
