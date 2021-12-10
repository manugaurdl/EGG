# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from itertools import repeat
from random import sample
from typing import Callable, List

import torch
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from torchvision.transforms.functional import crop


def extract_objs(
    imgs: List[torch.Tensor],
    bboxes: torch.Tensor,
    image_size: int,
) -> torch.Tensor:  # of size n_objs X 3 X H X W

    if len(imgs) == 1:
        bsz = len(bboxes)
        imgs = repeat(imgs[0], bsz)

    resizer = transforms.Resize(size=(image_size, image_size))
    segments = []
    for img, bbox in zip(imgs, bboxes):
        segments.append(resizer(crop(img, bbox[1], bbox[0], *bbox[2:])))
    return torch.stack(segments)


class BaseCollater:
    def __init__(
        self, max_objects: int, image_size: int, augmentations: Callable = None
    ):
        self.max_objs = max_objects
        if self.max_objs <= 2:
            raise RuntimeError(f"Max objs <=2 is not supported. Found {self.max_objs}")
        self.image_size = image_size
        self.augmentations = augmentations

    def __call__(self, batch):
        raise NotImplementedError


class RandomDistractorsCollater(BaseCollater):
    def __call__(self, batch):
        bsz = len(batch)
        mask = torch.zeros(bsz, self.max_objs)

        sender_objs, recv_objs, labels = [], [], []
        obj_ids, img_ids = [], []

        for elem_id, batch_elem in enumerate(batch):
            lineup = [batch_elem]

            dist_idx = 0
            for _ in range(self.max_objs - 1):
                if dist_idx == elem_id:
                    dist_idx = (dist_idx + 1) % bsz
                lineup.append(batch[dist_idx])
                dist_idx = (dist_idx + 1) % bsz

            batch_sender_imgs, batch_recv_imgs = [], []
            batch_bboxes, batch_labels = [], []
            batch_img_ids, batch_obj_ids = [], []
            for item in lineup:
                img = item[0]
                sender_img = self.augmentations(img) if self.augmentations else item[0]
                recv_img = self.augmentations(img) if self.augmentations else None

                batch_sender_imgs.append(sender_img)
                batch_recv_imgs.append(recv_img)

                obj_idx = random.randint(0, len(item[2]["bboxes"]) - 1)
                batch_bboxes.append(item[2]["bboxes"][obj_idx])
                batch_labels.append(item[1][obj_idx])
                batch_img_ids.append(item[2]["img_id"])
                batch_obj_ids.append(item[2]["obj_ids"][obj_idx])

            sender_objs.append(
                extract_objs(batch_sender_imgs, batch_bboxes, self.image_size)
            )
            if self.augmentations:
                recv_objs.append(
                    extract_objs(batch_recv_imgs, batch_bboxes, self.image_size)
                )

        sender_input = torch.stack(sender_objs)
        receiver_input = sender_input
        if self.augmentations:
            receiver_input = torch.stack(recv_objs)

        labels = torch.Tensor(labels)
        img_ids = torch.Tensor(img_ids)
        obj_ids = torch.Tensor(obj_ids)
        baselines = torch.Tensor([1 / self.max_objs] * len(batch))
        aux_input = {
            "baselines": baselines,
            "mask": mask,
            "img_ids": img_ids,
            "obj_ids": obj_ids,
        }
        return sender_input, labels, receiver_input, aux_input


class ContextualDistractorsCollater(BaseCollater):
    def __call__(self, batch):
        bsz = len(batch)
        mask = torch.zeros(bsz, self.max_objs)

        sender_objs, recv_objs, labels = [], [], []
        obj_ids, img_ids = [], []
        baselines = []
        for elem_id, elem in enumerate(batch):
            img = elem[0]
            sender_img = self.augmentations(img) if self.augmentations else img
            recv_img = self.augmentations(img) if self.augmentations else None

            all_bboxes = elem[2]["bboxes"]
            nb_objs_to_sample = min(len(all_bboxes), self.max_objs)
            begin_pad = nb_objs_to_sample
            mask[elem_id][begin_pad:] = 1.0

            obj_idxs = sample(range(len(all_bboxes)), k=nb_objs_to_sample)
            labels.append(torch.IntTensor([elem[1][idx] for idx in obj_idxs]))
            bboxes = [all_bboxes[idx] for idx in obj_idxs]

            sender_objs.append(extract_objs([sender_img], bboxes, self.image_size))
            if self.augmentations:
                recv_objs.append(extract_objs([recv_img], bboxes, self.image_size))

            baselines.append(torch.Tensor([1 / len(obj_idxs)]))
            img_ids.append(torch.Tensor(elem[2]["img_id"]))
            obj_ids.append(torch.Tensor([elem[2]["obj_ids"][idx] for idx in obj_idxs]))

        sender_input = pad_sequence(sender_objs, batch_first=True)
        receiver_input = sender_input
        if self.augmentations:
            receiver_input = pad_sequence(recv_objs, batch_first=True)
        labels = pad_sequence(labels, batch_first=True, padding_value=-1.0)

        img_ids = torch.cat(img_ids)
        obj_ids = pad_sequence(obj_ids, batch_first=True, padding_value=-1.0).squeeze()

        aux_input = {
            "baselines": torch.stack(baselines),
            "mask": mask,
            "img_ids": img_ids,
            "obj_ids": obj_ids,
        }
        return sender_input, labels, receiver_input, aux_input
