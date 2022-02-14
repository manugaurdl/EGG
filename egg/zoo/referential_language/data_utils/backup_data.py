# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license fod in the
# LICENSE file in the root directory of this source tree.

import json
import random
from math import floor
from pathlib import Path
from typing import Callable
from PIL import Image

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.transforms.functional import crop


def pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class VisualGenomeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_dir: str,
        metadata_dir: str,
        classes_path: str = "/private/home/rdessi/EGG/egg/zoo/referential_language/utils/classes_1600.txt",
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
                self.samples.append((img_path, objs))

        self.transform = transform
        self.max_objects = max_objects
        self.resizer = transforms.Resize(size=(image_size, image_size))

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

    def extract_object(self, image, obj_data):
        label = self.class2id[obj_data["names"][0]]
        y, x, h, w = obj_data["y"], obj_data["x"], obj_data["h"], obj_data["w"]
        # if we want to use augmentation maybe better to crop before transforming to tensor
        obj = self.resizer(crop(image, y, x, h, w))
        return obj, label

    def _load_and_transform(self, img_path):
        image = pil_loader(img_path)
        if self.transform:
            image = self.transform(image)
        return image

    def __getitem__(self, index):
        img_path, bboxes = self.samples[index]
        image = self._load_and_transform(img_path)

        cropped_obj, label = self.extract_object(image, bboxes[0])

        cropped_objs, labels = [cropped_obj], [label]
        distractors = random.sample(self.samples, k=self.max_objects - 1)
        for img_path, bboxes in distractors:
            image = self._load_and_transform(img_path)

            cropped_obj, label = self.extract_object(image, bboxes[0])
            labels.append(label)
            cropped_objs.append(cropped_obj)

        game_input = torch.stack(cropped_objs)
        labels = torch.Tensor(labels)

        mask = torch.ones(self.max_objects).bool()
        game_labels = torch.arange(self.max_objects)
        aux_input = {
            "mask": mask,
            "game_labels": game_labels,
            "baseline": torch.Tensor([1 / self.max_objects]),
        }

        return game_input, labels, torch.zeros(1), aux_input

    def __len__(self):
        return len(self.samples)


def get_dataloader(
    image_dir: str = "/datasets01/VisualGenome1.2/061517/",
    metadata_dir: str = "/private/home/rdessi/visual_genome/train_val_test_split_clean",
    batch_size: int = 32,
    split: str = "train",
    image_size: int = 32,
    max_objects: int = 20,
    seed: int = 111,
):
    transform = transforms.ToTensor()
    if split == "test":
        dataset = TestVisualGenomeDatasetRandomDistractors(
            image_dir,
            metadata_dir,
            transform=transform,
            max_objects=max_objects,
            image_size=image_size,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=6,
            pin_memory=True,
            drop_last=True,
        )

    dataset = VisualGenomeDataset(
        image_dir,
        metadata_dir,
        split=split,
        transform=transform,
        max_objects=max_objects,
        image_size=image_size,
    )
    sampler = None
    if dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=True, seed=seed)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=6,
        sampler=sampler,
        shuffle=(sampler is None),
        pin_memory=True,
        drop_last=True,
    )


class TestVisualGenomeDatasetRandomDistractors(IterableDataset):
    def __init__(
        self,
        image_dir: str,
        metadata_dir: str,
        classes_path: str = "/private/home/rdessi/EGG/egg/zoo/referential_language/utils/classes_1600.txt",
        transform: Callable = None,
        max_objects=20,
        image_size=64,
    ):
        path_images = Path(image_dir)
        path_metadata = Path(metadata_dir) / "test_objects.json"
        path_image_data = Path(metadata_dir) / "test_image_data.json"

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
                self.samples.append((img_path, objs))

        self.transform = transform
        self.max_objects = max_objects
        self.resizer = transforms.Resize(size=(image_size, image_size))

    def __iter__(self):
        self.curr_idx = 0

        print("len samples", len(self.samples), force=True)

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        per_gpu = int(floor(len(self.samples) / float(world_size)))

        print("per_gpu", per_gpu, force=True)

        iter_start = per_gpu * rank
        iter_end = iter_start + per_gpu

        worker_info = torch.utils.data.get_worker_info()
        if worker_info:  # num_workers is > 0
            per_worker = int(floor(per_gpu / worker_info.num_workers))
            iter_start = iter_start + worker_info.id * per_worker
            iter_end = iter_start + per_worker

        self.samples = self.samples[iter_start:iter_end]
        print(
            f"iter. gpu {rank}, worker {worker_info.id}",
            iter_start,
            iter_end,
            force=True,
        )
        self._load_new_sample()
        return self

    def __len__(self):
        return len(self.samples)

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

    def _extract_object(self, image, obj_data):
        label = self.class2id[obj_data["names"][0]]
        y, x, h, w = obj_data["y"], obj_data["x"], obj_data["h"], obj_data["w"]
        obj = self.resizer(crop(image, y, x, h, w))
        return obj, label

    def _load_new_sample(self):
        img_path, obj_data = self.samples[self.curr_idx]
        self.curr_img = self._load_and_transform(img_path)
        self.curr_obj_data = obj_data
        self.curr_obj_idx = 0

    def _load_and_transform(self, img_path):
        image = pil_loader(img_path)
        if self.transform:
            image = self.transform(image)
        return image

    def __next__(self):
        max_obj_idx = min(self.max_objects, len(self.curr_obj_data))

        if self.curr_obj_idx >= max_obj_idx:
            self.curr_obj_idx = 0
            self.curr_idx += 1
            if self.curr_idx >= len(self.samples):
                raise StopIteration
            self._load_new_sample()

        img = self.curr_img
        obj_data = self.curr_obj_data[self.curr_obj_idx]
        obj, label = self._extract_object(img, obj_data)
        self.curr_obj_idx += 1

        cropped_objs, labels = [obj], [label]
        distractors = random.sample(self.samples, k=self.max_objects - 1)
        for img_path, bboxes in distractors:
            image = self._load_and_transform(img_path)

            cropped_obj, label = self._extract_object(image, bboxes[0])
            labels.append(label)
            cropped_objs.append(cropped_obj)

        game_input = torch.stack(cropped_objs)
        labels = torch.Tensor(labels)

        mask = torch.ones(self.max_objects).bool()
        game_labels = torch.arange(self.max_objects)
        aux_input = {
            "mask": mask,
            "game_labels": game_labels,
            "baseline": torch.Tensor([1 / self.max_objects]),
        }

        return game_input, label, torch.zeros(1), aux_input
