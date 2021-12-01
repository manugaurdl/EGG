# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from random import sample
from typing import Callable, Iterable

import torch
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from torchvision.transforms.functional import crop


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
        dataset: Iterable = None,
        augmentations: Callable = None,
    ):
        super(RandomDistractorsCollater, self).__init__(
            max_objects, image_size, augmentations
        )
        if self.max_objs <= 2:
            raise RuntimeError(f"Max_objs <=2 is not supported. Found {self.max_objs}")
        self.dataset = dataset

    def __call__(self, batch):
        sender_segments, recv_segments, labels = [], [], []
        original_imgs, obj_ids, img_ids = [], [], []
        data_len, n_dis = len(self.dataset), self.max_objs - 1

        for elem in batch:
            batch_elem_sender, batch_elem_recv = [], []
            batch_labels, batch_obj_ids, batch_img_ids = [], [], []

            distractors = [self.dataset[i] for i in sample(range(data_len), k=n_dis)]
            for x in [elem] + distractors:
                img = x[0]
                idx = random.choice(range(len(x[2]["bboxes"])))
                img_bbox = x[2]["bboxes"][idx].unsqueeze(0)

                batch_labels.append(x[1][idx])
                batch_img_ids.append(x[2]["img_id"])
                batch_obj_ids.append(x[2]["obj_ids"][idx])

                img_sender = self.augmentations(img) if self.augmentations else img
                objs = extract_objs(img_sender, img_bbox, self.image_size)
                batch_elem_sender.append(objs)

                if self.augmentations:
                    img_recv = self.augmentations(img)
                    objs_recv = extract_objs(img_recv, img_bbox, self.image_size)
                    batch_elem_recv.append(objs_recv)

                original_imgs.append(transforms.functional.resize(img, size=(224, 224)))

            sender_segments.append(torch.cat(batch_elem_sender))
            if batch_elem_recv:
                recv_segments.append(torch.cat(batch_elem_recv))
            labels.append(torch.cat(batch_labels))

            img_ids.append(torch.cat(batch_img_ids))
            obj_ids.append(torch.cat(batch_obj_ids))

        sender_input = torch.stack(sender_segments)
        receiver_input = sender_input
        if recv_segments:
            receiver_input = torch.stack(recv_segments)
        labels = torch.stack(labels)

        img_ids = torch.stack(img_ids)
        obj_ids = torch.stack(obj_ids)
        original_imgs = torch.stack(original_imgs)

        return (
            sender_input,
            labels,
            receiver_input,
            {
                "baselines": torch.Tensor([1 / self.max_objs] * len(batch)),
                "mask": torch.zeros(len(batch)),
                "img_ids": img_ids,
                "obj_ids": obj_ids,
                "original_imgs": original_imgs,
            },
        )


class ContextualDistractorsCollater(BaseCollater):
    def __call__(self, batch):
        all_n_objs = [elem[2]["n_objs"] for elem in batch]
        max_objs = min(self.max_objs, max(all_n_objs))
        extra_objs = max_objs - torch.cat(all_n_objs)
        mask = torch.where(extra_objs > 0, extra_objs, torch.zeros(len(batch)))

        segments_sender, segments_recv, obj_labels = [], [], []
        baselines, original_imgs = [], []
        img_ids, obj_ids = [], []
        for elem in batch:
            bboxes = elem[2]["bboxes"]
            idxs = sample(range(len(bboxes)), k=min(len(bboxes), max_objs))
            obj_labels.append(elem[1][idxs])

            img, img_bboxes = elem[0], bboxes[idxs]
            img_sender = self.augmentations(img) if self.augmentations else img
            objs = extract_objs(img_sender, img_bboxes, self.image_size)
            segments_sender.append(objs)

            if self.augmentations:
                img_recv = self.augmentations(img)
                objs_recv = extract_objs(img_recv, img_bboxes, self.image_size)
                segments_recv.append(objs_recv)

            baselines.append(1 / objs.shape[0])
            img_ids.append(elem[2]["img_id"])
            obj_ids.append(elem[2]["obj_ids"][idxs])
            original_imgs.append(transforms.functional.resize(img, size=(224, 224)))

        sender_input = pad_sequence(segments_sender, batch_first=True)
        receiver_input = sender_input
        if segments_recv:
            receiver_input = pad_sequence(segments_recv, batch_first=True)
        labels = pad_sequence(obj_labels, batch_first=True)

        img_ids = torch.cat(img_ids)
        obj_ids = pad_sequence(obj_ids, batch_first=True)
        original_imgs = torch.stack(original_imgs)

        return (
            sender_input,
            labels,
            receiver_input,
            {
                "baselines": torch.Tensor(baselines),
                "mask": mask,
                "img_ids": img_ids,
                "obj_ids": obj_ids,
                "original_imgs": original_imgs,
            },
        )
