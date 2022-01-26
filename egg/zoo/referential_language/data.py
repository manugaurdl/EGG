# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license fod in the
# LICENSE file in the root directory of this source tree.

import json
from pathlib import Path
from typing import Callable
from PIL import Image

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.transforms.functional import crop


def pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class GaussianDataset:
    def __init__(self, image_size: int, max_objects: int, nb_samples: int):
        self.image_size = image_size
        self.nb_samples = nb_samples
        self.max_objects = max_objects

    def __len__(self):
        return self.nb_samples

    def __getitem__(self, index):
        x = torch.rand(self.max_objects, 3, self.image_size, self.image_size)
        labels = torch.ones(1)
        mask = torch.ones(self.max_objects)
        game_labels = torch.arange(self.max_objects)
        return x, labels, x, {"mask": mask, "game_labels": game_labels}


class VisualGenomeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_dir: str,
        metadata_dir: str,
        classes_path: str = "/private/home/rdessi/EGG/egg/zoo/referential_language/classes_1600.txt",
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

    def __getitem__(self, index):
        img_path, bboxes = self.samples[index]
        image = pil_loader(img_path)
        if self.transform:
            image = self.transform(image)

        cropped_objs, labels = [], []
        for obj in bboxes[: min(self.max_objects, len(bboxes))]:
            cropped_obj, label = self.extract_object(image, obj)
            labels.append(label)
            cropped_objs.append(cropped_obj)

        agent_input = torch.stack(cropped_objs)
        labels = torch.Tensor(labels)
        return agent_input, labels, None, len(agent_input)

    def __len__(self):
        return len(self.samples)


class Collater:
    def __init__(self, max_objects):
        self.max_objects = max_objects

    def __call__(self, batch):
        max_objs = min(self.max_objects, max([x[3] for x in batch]))
        inp, lab = [], []
        game_labels, mask, baseline = [], [], []
        for x, l, _, num_elems in batch:
            inp.append(x)
            lab.append(l)

            x = torch.zeros(max_objs)
            x[:num_elems] = 1
            mask.append(x)
            game_labels.append(torch.arange(self.max_objects))
            baseline.append(torch.Tensor([1 / num_elems]))

        inp = torch.nn.utils.rnn.pad_sequence(inp, batch_first=True)
        label = torch.nn.utils.rnn.pad_sequence(lab, batch_first=True, padding_value=-1)
        mask = torch.stack(mask)
        game_labels = torch.stack(game_labels)
        baseline = torch.stack(baseline)

        return (
            inp,
            label,
            None,
            {"mask": mask, "game_labels": game_labels, "baseline": baseline},
        )


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
        collate_fn=Collater(max_objects),
        shuffle=(sampler is None),
        pin_memory=True,
        drop_last=True,
    )


def get_gaussian_dataloader(image_size, max_objects, batch_size, **kwargs):
    return torch.utils.data.DataLoader(
        GaussianDataset(image_size, max_objects, nb_samples=2_000),
        batch_size=batch_size,
        num_workers=12,
        pin_memory=True,
        drop_last=True,
    )
