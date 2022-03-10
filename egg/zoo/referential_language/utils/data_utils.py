# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license fod in the
# LICENSE file in the root directory of this source tree.

import random
from collections import defaultdict
from PIL import ImageFilter

import torch
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence


def collate(batch):
    sender_input, labels, recv_input = [], [], []
    aux_input = defaultdict(list)
    for obj_sender, label, obj_recv, aux in batch:
        sender_input.append(obj_sender)
        labels.append(label)
        recv_input.append(obj_recv)

        for k, v in aux.items():
            aux_input[k].append(v)

    def pad(elem):
        if isinstance(elem, list):
            return pad_sequence(elem, batch_first=True, padding_value=-1)
        elif isinstance(elem, dict):
            return {k: pad(v) for k, v in elem.items()}
        elif isinstance(elem, torch.Tensor):
            return elem
        else:
            raise RuntimeError("Cannot pad elem of type {type(elem)}")

    sender_input = pad(sender_input)
    recv_input = pad(recv_input)
    labels = pad(labels)
    aux_input = pad(aux_input)
    aux_input["mask"] = sender_input[:, :, 0, 0, 0] != -1
    return sender_input, labels, recv_input, aux_input


class GaussianBlur:
    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, img):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        return img.filter(ImageFilter.GaussianBlur(radius=sigma))


class ImageTransformation:
    def __init__(self, use_augmentation: bool = False):
        transformations = [transforms.ToTensor()]
        if use_augmentation:
            augmentations = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            ]
            transformations = augmentations + transformations

        self.transform = transforms.Compose(transformations)

    def __call__(self, x):
        return self.transform(x)


"""
class MyBatch(Batch):
    def to(self, device: torch.device):
        self.sender_input = move_to(self.sender_input, device)
        self.labels = move_to(self.labels, device)
        self.receiver_input = move_to(self.receiver_input, device)
        # all fields of aux_input except mask are not needed during trying,
        # we store them on cpu to save gpu space
        self.aux_input = move_to(self.aux_input, torch.device("cpu"))
        self.aux_input["mask"] = move_to(self.aux_input["mask"], device)
        return self
"""
