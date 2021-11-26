# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import json
import random
from pathlib import Path
from random import sample
from typing import Callable, Iterable
from PIL import Image, ImageFilter

import torch
import torchvision
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.transforms.functional import crop


def pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class VisualGenomeDataset(torchvision.datasets.VisionDataset):
    def __init__(self, image_dir, metadata_dir, split="train", transform=None):
        super(VisualGenomeDataset, self).__init__(root=image_dir, transform=transform)
        path_images = Path(image_dir)
        assert split in ["train", "val"], f"Unknown dataset split: {split}"
        path_objects = Path(metadata_dir) / f"{split}_objects.json"
        path_image_data = Path(metadata_dir) / f"{split}_image_data.json"

        with open(path_image_data) as fin:
            img_data = json.load(fin)
        with open(path_objects) as fin:
            obj_data = json.load(fin)

        self.samples = []
        for img, objs in zip(img_data, obj_data):
            # transforming url to local path
            # url is of this form: https://cs.stanford.edu/people/rak248/VG_100K_2/1.jpg
            # and will become {path_to_vg_folder}VG_100K_2/1.jpg
            img_path = path_images / "/".join(img["url"].split("/")[-2:])
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
            if w == 1 or h == 1:
                continue
            if (x + w) * (y + h) / (img_w * img_h) > 0.01:
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
    return torch.stack(segments)  # between  72064 and 72191 have w or h = 1


class BaseCollater:
    def __init__(
        self, max_objects: int, image_size: int, augmentations: Callable = None
    ):
        self.max_objs = max_objects
        self.image_size = image_size
        self.augmentations = augmentations

    def __call__(self, batch):
        raise NotImplementedError


class RandomDistractorsCollater(BaseCollater):
    def __init__(
        self,
        max_objects: int,
        image_size: int,
        augmentations: Callable = None,
        dataset: Iterable = None,
    ):
        super(RandomDistractorsCollater, self).__init__(
            max_objects, image_size, augmentations
        )
        if self.max_objects <= 2:
            raise RuntimeError(f"Max_objs <=2 is not supporte. Found {max_objects}")
        self.dataset = dataset

    def __call__(self, batch):
        sender_segments, recv_segments = [], []
        data_len, n_dis = len(self.dataset), self.max_objs - 1
        for elem in batch:
            batch_elem_sender, batch_elem_recv = [], []
            distractors = [self.dataset[i] for i in sample(range(data_len), k=n_dis)]
            for x in [elem] + distractors:
                img, img_bboxes = x[0], random.choice(x[2]["bboxes"]).unsqueeze(0)

                img_sender = self.augmentations(img) if self.augmentations else img
                objs = extract_objs(img_sender, img_bboxes, self.image_size)
                batch_elem_sender.append(objs)

                if self.augmentations:
                    img_recv = self.augmentasion(img)
                    objs_recv = extract_objs(img_recv, img_bboxes, self.image_size)
                    batch_elem_recv.append(objs_recv)

            sender_segments.append(torch.cat(batch_elem_sender))
            if batch_elem_recv:
                recv_segments.append(torch.cat(batch_elem_recv))

        sender_input = torch.stack(sender_segments)
        receiver_input = sender_input
        if recv_segments:
            receiver_input = torch.stack(recv_segments)

        return (
            sender_input,
            torch.IntTensor([1]),  # dummy label
            receiver_input,
            {
                "baselines": torch.Tensor([1 / self.max_objs] * len(batch)),
                "mask": torch.zeros(len(batch)),
            },
        )


class ContextualDistractorsCollater(BaseCollater):
    def __call__(self, batch):
        all_n_objs = [elem[2]["n_objs"] for elem in batch]
        max_objs = min(self.max_objs, max(all_n_objs))
        extra_objs = max_objs - torch.cat(all_n_objs)
        mask = torch.where(extra_objs > 0, extra_objs, torch.zeros(len(batch)))

        segments_sender, segments_recv = [], []
        all_bboxes, baselines = [], []
        for elem in batch:
            bboxes = elem[2]["bboxes"]
            bboxes_idx = sample(range(len(bboxes)), k=min(len(bboxes), max_objs))

            img, img_bboxes = elem[0], bboxes[bboxes_idx]
            img_sender = self.augmentations(img) if self.augmentations else img
            objs = extract_objs(img_sender, img_bboxes, self.image_size)
            segments_sender.append(objs)

            if self.augmentations:
                img_recv = self.augmentations(img)
                objs_recv = extract_objs(img_recv, img_bboxes, self.image_size)
                segments_recv.append(objs_recv)

            baselines.append(1 / objs.shape[0])
            all_bboxes.append(img_bboxes)

        sender_input = pad_sequence(segments_sender, batch_first=True)
        receiver_input = sender_input
        if segments_recv:
            receiver_input = pad_sequence(segments_recv, batch_first=True)
        batched_bboxes = pad_sequence(all_bboxes, batch_first=True)

        return (
            sender_input,
            torch.IntTensor([1]),  # dummy label
            receiver_input,
            {
                "bboxes": batched_bboxes,
                "mask": mask,
                "baselines": torch.Tensor(baselines),
            },
        )


class GaussianBlur:
    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        return x.filter(ImageFilter.GaussianBlur(radius=sigma))


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

    augmentations = None
    if use_augmentations:
        color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        transformations = [
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
        ]
        augmentations = transforms.Compose(transformations)

    if contextual_distractors:
        collater = ContextualDistractorsCollater(
            max_objects, image_size, augmentations=augmentations
        )
    else:
        collater = RandomDistractorsCollater(
            max_objects, image_size, dataset, augmentations=augmentations
        )

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
