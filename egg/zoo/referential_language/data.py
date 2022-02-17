# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license fod in the
# LICENSE file in the root directory of this source tree.

import json
from pathlib import Path
from typing import Callable
from PIL import Image

import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.transforms.functional import crop, resize


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

    def __len__(self):
        return len(self.samples)

    def _load_and_transform(self, img_path):
        image = pil_loader(img_path)
        if self.transform:
            image = self.transform(image)
        return image

    def _crop_and_resize_object(self, image, obj_item):
        y, x, h, w = obj_item["y"], obj_item["x"], obj_item["h"], obj_item["w"]
        return self.resizer(crop(image, y, x, h, w))

    def __getitem__(self, index):
        img_path, bboxes = self.samples[index]

        sender_image = self._load_and_transform(img_path)
        recv_image = self._load_and_transform(img_path)

        sender_objs, labels, recv_objs = [], [], []
        for obj_item in bboxes[: min(self.max_objects, len(bboxes))]:
            sender_obj = self._crop_and_resize_object(sender_image, obj_item)
            recv_obj = self._crop_and_resize_object(recv_image, obj_item)

            sender_objs.append(sender_obj)
            recv_objs.append(recv_obj)

            label = next(filter(lambda n: n in self.class2id, obj_item["names"]), None)
            assert label is not None
            labels.append(self.class2id[label])

        sender_input = torch.stack(sender_objs)
        recv_input = torch.stack(recv_objs)
        labels = torch.Tensor(labels)

        sender_image = resize(sender_image, size=(128, 128))
        recv_image = resize(recv_image, size=(128, 128))

        return sender_input, labels, recv_input, sender_image, recv_image


def collate(batch):
    sender_input, labels, recv_input = [], [], []
    sender_images, recv_images = [], []
    for obj_sender, label, obj_recv, sender_image, recv_image in batch:
        sender_input.append(obj_sender)
        labels.append(label)
        recv_input.append(obj_recv)

        sender_images.append(sender_image)
        recv_images.append(recv_image)

    sender_input = pad_sequence(sender_input, batch_first=True, padding_value=-1)
    recv_input = pad_sequence(recv_input, batch_first=True, padding_value=-1)
    labels = pad_sequence(labels, batch_first=True, padding_value=-1)

    mask = sender_input[:, :, 0, 0, 0] != -1
    baseline = 1 / mask.int().sum(-1)
    bsz, max_objs = sender_input.shape[:2]
    game_labels = torch.arange(max_objs).repeat(bsz, 1)
    aux_input = {
        "mask": mask,
        "game_labels": game_labels,
        "baseline": baseline,
        "sender_images": torch.stack(sender_images),
        "recv_images": torch.stack(recv_images),
    }
    return sender_input, labels, recv_input, aux_input


def get_dataloader(
    image_dir: str = "/private/home/rdessi/visual_genome",
    metadata_dir: str = "/private/home/rdessi/visual_genome/filtered_splits",
    batch_size: int = 32,
    split: str = "train",
    image_size: int = 32,
    max_objects: int = 20,
    seed: int = 111,
):
    ds = VisualGenomeDataset(
        image_dir=image_dir,
        metadata_dir=metadata_dir,
        split=split,
        max_objects=max_objects,
        image_size=image_size,
    )

    sampler = None
    if dist.is_initialized():
        shuffle = split != "test"
        sampler = DistributedSampler(ds, shuffle=shuffle, drop_last=True, seed=seed)

    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=6,
        sampler=sampler,
        collate_fn=collate,
        shuffle=(sampler is None),
        pin_memory=True,
        drop_last=True,
    )
