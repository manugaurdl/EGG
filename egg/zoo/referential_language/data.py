#!/usr/bin/env python
# coding: utf-8


import json
import re
from pathlib import Path
from PIL import Image
from typing import List

import torch
import torchvision
from torchvision import transforms


def resize_bboxes(
    self, boxes: torch.Tensor, new_size: List[int], original_size: List[int]
) -> torch.Tensor:

    ratios = [
        torch.tensor(s, dtype=torch.float32, device=boxes.device)
        / torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)

    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)


class VisualGenomeDataset(torchvision.datasets.VisionDataset):
    def __init__(self, dataset_dir, transform=None):
        super(VisualGenomeDataset, self).__init__(root=dataset_dir, transform=transform)

        self.path = Path(dataset_dir)
        path_objects = self.path / "objects.json"
        path_image_data = self.path / "image_data.json"

        with open(path_objects) as fin:
            self.data = json.load(fin)[:1]
        with open(path_image_data) as fin:
            self.metadata = json.load(fin)[:1]

        all_obj_names = []
        for i in self.data:
            for j in i["objects"]:
                if j["synsets"] != []:
                    all_obj_names.append(j["synsets"][0])
        else:
            all_obj_names.append("No synset")
        dic_names = dict(enumerate(set(all_obj_names)))
        self.dic_names = {v: k for k, v in dic_names.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        def pil_loader(path: str) -> Image.Image:
            with open(path, "rb") as f:
                img = Image.open(f)
                return img.convert("RGB")

        zipped1 = [[list(i.values())[0], list(i.values())[1]] for i in self.data]
        image_id, objects_info = zip(*zipped1)

        zipped2 = [
            [j["url"], j["width"], j["height"]]
            for j in self.metadata
            for i in image_id
            if j["image_id"] == i
        ]
        image_url, image_w, image_h = zip(*zipped2)

        zipped3 = [
            [
                j["names"],
                torch.Tensor([self.dic_names[z] for z in j["synsets"]]).long(),
                torch.Tensor((j["x"], j["y"], j["w"], j["h"])),
            ]
            for j in objects_info[index]
        ]
        obj_names, obj_labels, obj_xywh = zip(*zipped3)

        im_path = self.p / "Images" / re.findall("VG_.*.jpg", image_url[index])[0]

        image = pil_loader(im_path)

        if self.transform:
            sender_input, receiver_input = self.transform(image), self.transform(image)

        if self.target_transform:
            obj_xywh_t = self.target_transform(obj_xywh, (image.size[1], image.size[0]))

        return (
            sender_input,
            torch.cat(obj_labels),
            receiver_input,
            torch.cat(obj_xywh_t),
        )


def get_dataloader(
    dataset_dir: str,
    batch_size: int = 32,
    image_size: int = 32,
    is_distributed: bool = False,
    seed: int = 111,
):

    simple_transf = transforms.Compose(
        [
            transforms.Resize(size=(image_size, image_size)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = VisualGenomeDataset(root=dataset_dir, transform=simple_transf)

    if is_distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True, seed=seed
        )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    return loader
