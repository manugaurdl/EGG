# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from random import sample
from typing import Callable, List, NamedTuple, Optional

import torch
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from torchvision.transforms.functional import crop


class BatchElem(NamedTuple):
    img: torch.Tensor
    bbox: List[float]
    label: int
    obj_id: int
    img_id: int


class Collater:
    def __init__(
        self,
        max_objects: int,
        image_size: int,
        contextual_distractors: bool,
        create_recv_input: bool,
        augmentations: Callable = None,
    ):
        self.max_objs = max_objects
        if self.max_objs <= 2:
            raise RuntimeError(f"Max objs <=2 is not supported. Found {self.max_objs}")
        self.image_size = image_size
        self.contextual_distractors = contextual_distractors
        self.augmentations = augmentations
        self.create_recv_input = create_recv_input

    @staticmethod
    def extract_obj(
        img: torch.Tensor,
        bbox: List[int],
        image_size: int,
    ) -> torch.Tensor:  # of size 1 X 3 X H X W
        resizer = transforms.Resize(size=(image_size, image_size))
        return resizer(crop(img, bbox[1], bbox[0], *bbox[2:]))

    def _create_batch_elem(self, lineup):
        sender_objs, recv_objs, labels = [], [], []
        img_ids, obj_ids = [], []
        img = lineup[0].img
        if self.augmentations:
            img = self.augmentations(img)
        for item in lineup:
            img_ids.append(item.img_id)
            obj_ids.append(item.obj_id)
            labels.append(item.label)
            if not self.contextual_distractors:
                img = self.augmentations(item.img) if self.augmentations else item.img

            sender_objs.append(self.extract_obj(img, item.bbox, self.image_size))
            if self.create_recv_input:
                recv_objs.append(self.extract_obj(img, item.bbox, self.image_size))

        sender_objs = torch.stack(sender_objs)
        if recv_objs:
            torch.stack(recv_objs)
        labels = torch.Tensor(labels)
        img_ids = torch.Tensor(img_ids)
        obj_ids = torch.Tensor(obj_ids)
        baseline = torch.Tensor([1 / len(lineup)])
        mask = torch.zeros(self.max_objs)
        begin_pad = min(len(lineup), self.max_objs)
        mask[begin_pad:] = 1.0
        return sender_objs, recv_objs, labels, img_ids, obj_ids, mask, baseline

    def _pack_candidate(self, item, obj_idx):
        img = item[0]
        label = item[1][obj_idx]
        img_id = item[2]["img_id"]
        bbox = item[2]["bboxes"][obj_idx]
        obj_id = item[2]["obj_ids"][obj_idx]
        return BatchElem(img, bbox, label, img_id, obj_id)

    def _pack_agent_input(
        self,
        sender_objs: List[torch.Tensor],
        receiver_objs: Optional[List[torch.Tensor]],
        labels: List[torch.Tensor],
        img_ids: List[torch.Tensor],
        obj_ids: List[torch.Tensor],
        baselines: List[torch.Tensor],
        mask: torch.Tensor,
    ):
        sender_input = pad_sequence(sender_objs, batch_first=True)
        receiver_input = sender_input
        if receiver_objs:
            receiver_input = pad_sequence(receiver_objs, batch_first=True)

        labels = pad_sequence(labels, batch_first=True, padding_value=-1.0)
        obj_ids = pad_sequence(obj_ids, batch_first=True, padding_value=-1.0).squeeze()
        img_ids = torch.cat(img_ids)

        aux_input = {
            "baselines": torch.cat(baselines),
            "mask": torch.stack(mask),
            "img_ids": img_ids,
            "obj_ids": obj_ids,
        }
        return sender_input, labels, receiver_input, aux_input

    def __call__(self, batch):
        sender_objs, recv_objs, labels = [], [], []
        obj_ids, img_ids = [], []
        masks, baselines = [], []
        for elem_id, elem in enumerate(batch):
            if self.contextual_distractors:
                all_bboxes = elem[2]["bboxes"]
                nb_objs_to_sample = min(len(all_bboxes), self.max_objs)
                obj_idxs = sample(range(len(all_bboxes)), k=nb_objs_to_sample)
                lineup = [self._pack_candidate(elem, idx) for idx in obj_idxs]
            else:
                bsz = len(batch)
                lineup = []
                obj_idx = random.randint(0, len(elem[2]["bboxes"]) - 1)
                lineup.append(self._pack_candidate(elem, obj_idx))

                dist_idx = random.randint(0, bsz - 1)
                for _ in range(self.max_objs - 1):
                    if dist_idx == elem_id:
                        dist_idx = (dist_idx + 1) % bsz
                    candidate = batch[dist_idx]
                    obj_idx = random.randint(0, len(candidate[2]["bboxes"]) - 1)
                    lineup.append(self._pack_candidate(candidate, obj_idx))
                    dist_idx = (dist_idx + 1) % bsz

            batched_elem = self._create_batch_elem(lineup)
            batched_sender_objs, batched_recv_objs, batched_labels = batched_elem[:3]
            batched_img_ids, batched_obj_ids, mask, baseline = batched_elem[3:]

            sender_objs.append(batched_sender_objs)
            labels.append(batched_labels)
            img_ids.append(batched_img_ids)
            obj_ids.append(batched_obj_ids)
            baselines.append(baseline)
            masks.append(mask)

        agent_input = self._pack_agent_input(
            sender_objs,
            recv_objs,
            labels,
            img_ids,
            obj_ids,
            baselines,
            masks,
        )
        return agent_input
