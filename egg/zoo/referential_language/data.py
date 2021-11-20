#!/usr/bin/env python
# coding: utf-8


import json
import logging
import random
from pathlib import Path
from PIL import Image

import torch
import torchvision
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from torchvision.transforms.functional import crop

logger = logging.getLogger(__name__)


def pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class VisualGenomeDataset(torchvision.datasets.VisionDataset):
    def __init__(self, dataset_dir, transform=None):
        super(VisualGenomeDataset, self).__init__(root=dataset_dir, transform=transform)
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
            image = self.transform(image)

        boxes = []
        for x, y, h, w in obj_info:
            if (x + w) * (y + h) / (img_w * img_h) > 0.01:
                # bboxes format: xmin, ymax, width, height
                boxes.append(torch.IntTensor([x, y, h, w]))

        if len(boxes) <= 1:
            return self.__getitem__(random.randint(0, len(self) - 1))
        else:
            return (
                image,
                torch.LongTensor([1] * len(obj_info)),  # dummy category per object
                {"bboxes": torch.stack(boxes), "n_objs": torch.Tensor([len(boxes)])},
            )


def extract_objs(
    img: torch.Tensor,
    bboxes: torch.Tensor,
    image_size: int,
) -> torch.Tensor:
    # returns a Tensor of size n_objs X 3 X H X W
    resizer = transforms.Resize(size=(image_size, image_size))
    segments = [resizer(crop(img, bbox[1], bbox[0], *bbox[2:])) for bbox in bboxes]
    return torch.stack(segments)


class Collater:
    def __init__(self, max_objects: int, image_size: int):
        self.max_objs = max_objects
        self.image_size = image_size

    def __call__(self, batch):
        n_objs = self.max_objs - torch.cat([elem[2]["n_objs"] for elem in batch])
        mask = torch.where(n_objs > 0, n_objs, torch.zeros(len(batch)))

        segments, bboxes, labels = [], [], []
        for elem in batch:
            bboxes = elem[2]["bboxes"][: self.max_objs]
            segments.append(
                extract_objs(img=elem[0], bboxes=bboxes, image_size=self.image_size)
            )
            bboxes.append(elem[2]["bboxes"][: self.max_objs])
            labels.append(elem[1][: self.max_objs])

        sender_input = torch.nn.utils.rnn.pad_sequence(segments, batch_first=True)
        labels = pad_sequence(labels)
        batched_bboxes = pad_sequence(bboxes, batch_first=True)
        receiver_input = sender_input

        breakpoint()
        return (
            sender_input,
            labels,
            receiver_input,
            {"bboxes": batched_bboxes, "mask": mask},
        )


def get_dataloader(
    dataset_dir: str = "/datasets01/VisualGenome1.2/061517",
    batch_size: int = 32,
    image_size: int = 32,
    max_objects: int = 20,
    seed: int = 111,
):
    dataset = VisualGenomeDataset(
        dataset_dir=dataset_dir, transform=transforms.ToTensor()
    )

    collater = Collater(max_objects=max_objects, image_size=image_size)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collater,
        num_workers=0,  # TODO TO CHANGE THIS
        pin_memory=True,
        drop_last=True,
    )
    return loader
