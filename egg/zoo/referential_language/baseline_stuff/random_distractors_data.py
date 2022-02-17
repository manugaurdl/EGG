# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license fod in the
# LICENSE file in the root directory of this source tree.

import json
from math import floor
from pathlib import Path
from random import sample
from typing import Callable
from PIL import Image

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset
from torchvision import transforms
from torchvision.transforms.functional import crop


def pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class VisualGenomeDataset:
    def __init__(
        self,
        image_dir: str,
        metadata_dir: str,
        classes_path: str = "/private/home/rdessi/EGG/egg/zoo/referential_language/utils/classes_1600.txt",
        split: str = "train",
        transform: Callable = transforms.ToTensor(),
        max_objects=10,
        image_size=64,
    ):
        assert max_objects >= 3
        path_images = Path(image_dir)
        path_metadata = Path(metadata_dir) / f"{split}_objects.json"
        path_image_data = Path(metadata_dir) / f"{split}_image_data.json"

        with open(path_image_data) as img_in, open(path_metadata) as metadata_in:
            img_data, object_data = json.load(img_in), json.load(metadata_in)
        assert len(img_data) == len(object_data)

        self.class2id = {}
        idx = 0
        with open(classes_path) as f:
            for line in f:
                names = line.strip().split(",")
                for name in names:
                    self.class2id[name] = idx
                    idx += 1

        object_dict = {}
        for object_item in object_data:
            object_dict[object_item["image_id"]] = object_item

        self.samples = []
        for img_item in img_data:
            image_id = img_item["image_id"]
            object_item = object_dict[image_id]

            img_path = path_images / "/".join(img_item["url"].split("/")[-2:])

            self.samples.append((img_path, object_item["objects"]))

        self.id2class = {v: k for k, v in self.class2id.items()}
        self.transform = transform
        self.max_objects = max_objects
        self.resizer = transforms.Resize(size=(image_size, image_size))

    def _extract_object(self, image, obj_data):
        label = None
        for name in obj_data["names"]:
            if name in self.class2id:
                label = self.class2id[name]
                break
        assert label is not None

        y, x, h, w = obj_data["y"], obj_data["x"], obj_data["h"], obj_data["w"]
        obj = self.resizer(crop(image, y, x, h, w))
        return obj, label

    def __len__(self):
        return len(self.samples)

    def _load_and_transform(self, img_path):
        image = pil_loader(img_path)
        if self.transform:
            image = self.transform(image)
        return image

    def _get_aux_input_random_dist(self):
        mask = torch.ones(self.max_objects).bool()
        game_labels = torch.arange(self.max_objects)
        baseline = torch.Tensor([1 / self.max_objects])
        return {"mask": mask, "game_labels": game_labels, "baseline": baseline}


class TrainVisualGenomeDatasetRandomDistractors(VisualGenomeDataset):
    def __getitem__(self, index):
        labels, cropped_objs = [], []
        lineup = [self.samples[index]] + sample(self.samples, k=self.max_objects - 1)
        for img_path, bboxes in lineup:
            image = self._load_and_transform(img_path)

            cropped_obj, label = self._extract_object(image, bboxes[0])
            labels.append(label)
            cropped_objs.append(cropped_obj)

        game_input = torch.stack(cropped_objs)
        labels = torch.Tensor(labels)
        aux_input = self._get_aux_input_random_dist()
        return game_input, labels, torch.zeros(1), aux_input


class TestVisualGenomeDatasetRandomDistractors(VisualGenomeDataset, IterableDataset):
    def __init__(self, *args, **kwargs):
        super(TestVisualGenomeDatasetRandomDistractors, self).__init__(*args, **kwargs)

    def __iter__(self):
        self.curr_idx = 0
        self.curr_obj_idx = 0

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        per_gpu = int(floor(len(self.samples) / float(world_size)))

        iter_start = per_gpu * rank
        iter_end = iter_start + per_gpu

        worker_info = torch.utils.data.get_worker_info()
        if worker_info:  # num_workers is > 0
            per_worker = int(floor(per_gpu / worker_info.num_workers))
            iter_start = iter_start + worker_info.id * per_worker
            iter_end = iter_start + per_worker

        self.samples = self.samples[iter_start:iter_end]
        self._load_new_sample()  # load first sample
        return self

    def _load_new_sample(self):
        img_path, obj_data = self.samples[self.curr_idx]
        self.curr_img = self._load_and_transform(img_path)
        self.curr_obj_data = obj_data
        self.max_obj_idx = min(self.max_objects, len(self.curr_obj_data))

    def __next__(self):
        if self.curr_obj_idx >= self.max_obj_idx:
            self.curr_obj_idx = 0
            self.curr_idx += 1
            if self.curr_idx >= len(self.samples):
                raise StopIteration
            self._load_new_sample()

        obj_data = self.curr_obj_data[self.curr_obj_idx]
        obj, label = self._extract_object(self.curr_img, obj_data)
        self.curr_obj_idx += 1

        cropped_objs, labels = [obj], [label]
        distractors = sample(self.samples, k=self.max_objects - 1)
        for img_path, bboxes in distractors:
            image = self._load_and_transform(img_path)

            cropped_obj, label = self._extract_object(image, bboxes[0])
            labels.append(label)
            cropped_objs.append(cropped_obj)

        game_input = torch.stack(cropped_objs)
        labels = torch.Tensor(labels)

        aux_input = self._get_aux_input_random_dist()
        return game_input, label, torch.zeros(1), aux_input
