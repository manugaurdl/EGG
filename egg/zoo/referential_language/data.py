#!/usr/bin/env python
# coding: utf-8


import json
import logging
from pathlib import Path
from typing import Callable, List
from PIL import Image

import torch
import torchvision
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from torchvision.transforms.functional import crop

logger = logging.getLogger(__name__)


class VisualGenomeDataset(torchvision.datasets.VisionDataset):
    def __init__(self, dataset_dir, transform=None, target_transform=None):
        super(VisualGenomeDataset, self).__init__(
            root=dataset_dir, transform=transform, target_transform=target_transform
        )
        path = Path(dataset_dir)
        path_objects, path_image_data = path / "objects.json", path / "image_data.json"

        with open(path_image_data) as fin:
            img_data = json.load(fin)
        with open(path_objects) as fin:
            obj_data = json.load(fin)

        self.samples = []
        for img, objs in zip(img_data, obj_data):
            # transforming url to local path
            # url is of this form: https://cs.stanford.edu/people/rak248/VG_100K_2/1.jpg
            # and will become {path_to_vg_folder}VG_100K_2/1.jpg
            img_path = path / "/".join(img["url"].split("/")[-2:])
            h, w = img["height"], img["width"]
            obj_info = [
                (obj["x"], obj["y"], obj["h"], obj["w"]) for obj in objs["objects"]
            ]
            self.samples.append((img_path, h, w, obj_info))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, img_h, img_w, obj_info = self.samples[index]

        image = pil_loader(img_path)
        if self.transform:
            sender_input, receiver_input = self.transform(image), self.transform(image)

        boxes = torch.stack(
            [torch.IntTensor([x, y, x + w, y + h]) for x, y, h, w in obj_info]
        )
        if self.target_transform:
            boxes = self.target_transform(
                boxes=boxes, original_size=torch.IntTensor([img_w, img_h])
            )

        return (
            sender_input,
            torch.LongTensor([1] * len(obj_info)),  # dummy category per object
            receiver_input,
            {"boxes": boxes, "n_objs": boxes.shape[0]},
        )


def crop_and_augment(
    img: torch.Tensor, coords: torch.Tensor, transform: Callable = None
):
    top, left = int(coords[2]), int(coords[0])
    height = max(1, int(coords[3] - coords[2]))
    width = max(1, int(coords[1] - coords[0]))
    return crop(img, top=top, left=left, height=height, width=width)


def get_bboxes(
    img: torch.Tensor, bboxes: torch.Tensor, transform: Callable = None
) -> torch.Tensor:
    # returns a Tensor of size n_objs X 3 X H X W
    return torch.stack([crop_and_augment(img, bbox, transform) for bbox in bboxes])


class Collater:
    def __init__(self, max_objects, transform=None):
        self.max_objects = max_objects
        self.transform = transform

    def __call__(self, batch):
        breakpoint()
        mask_len = [max(0, self.max_objects - elem[3]["n_objs"]) for elem in batch]
        mask_len = torch.Tensor(mask_len)

        boxes = pad_sequence(
            [elem[3]["boxes"][: self.max_objects] for elem in batch], batch_first=True
        )
        segments = [
            get_bboxes(elem[0], elem[3]["n_objs"], transform=self.transform)
            for elem in batch
        ]

        sender_input = torch.nn.utils.rnn.pad_sequence(segments, batch_first=True)

        receiver_input = torch.nn.utils.rnn.pad_sequence(sender_input, batch_first=True)

        labels = pad_sequence([elem[1][: self.max_objects] for elem in batch])

        return (
            sender_input,
            labels,
            receiver_input,
            {"boxes": boxes, "mask_len": mask_len},
        )


def pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def get_dataloader(
    dataset_dir: str = "/datasets01/VisualGenome1.2/061517",
    batch_size: int = 32,
    image_size: int = 32,
    max_objects: int = 20,
    seed: int = 111,
):

    transformations = [
        transforms.Resize(size=(image_size, image_size)),
        transforms.ToTensor(),
    ]

    dataset = VisualGenomeDataset(
        dataset_dir=dataset_dir, transform=transforms.Compose(transformations)
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=Collater(max_objects=max_objects),
        num_workers=0,  # TODO TO CHANGE THIS
        pin_memory=True,
        drop_last=True,
    )
    return loader


class BboxResizer:
    def __init__(self, new_size: List[int]):
        self.new_size = torch.IntTensor(new_size)

    def __call__(
        self, boxes: torch.Tensor, original_size: torch.Tensor
    ) -> torch.Tensor:
        ratios = self.new_size / original_size

        xmin, ymin, xmax, ymax = boxes.unbind(1)
        ratio_width, ratio_height = ratios.unbind(0)

        xmin = xmin * ratio_width
        xmax = xmax * ratio_width
        ymin = ymin * ratio_height
        ymax = ymax * ratio_height
        return torch.stack((xmin, ymin, xmax, ymax), dim=1)
