# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import Any, List

import torch
from PIL import ImageFilter
from torchvision import datasets, transforms


def collate_with_random_recv_input(batch: List[Any]):
    sender_input, receiver_input, class_labels = [], [], []
    for elem in batch:
        sender_input.append(elem[0][0])
        receiver_input.append(elem[0][1])
        class_labels.append(torch.LongTensor([elem[1]]))

    bsz = len(batch)
    sender_input = torch.stack(sender_input)
    receiver_input = torch.stack(receiver_input)
    class_labels = torch.stack(class_labels).view(2, 1, bsz // 2, -1)

    random_order = torch.stack([torch.randperm(bsz // 2) for _ in range(2)])
    target_position = torch.argmin(random_order, dim=1)

    return (
        sender_input,
        class_labels,
        receiver_input,
        {"target_position": target_position, "random_order": random_order},
    )


def collate(batch: List[Any]):
    sender_input, labels, receiver_input = [], [], []
    for elem in batch:
        sender_input.append(elem[0][0])
        receiver_input.append(elem[0][1])
        labels.append(torch.LongTensor([elem[1]]))

    sender_input = torch.stack(sender_input)
    receiver_input = torch.stack(receiver_input)
    labels = torch.stack(labels)

    return sender_input, labels, receiver_input


def get_dataloader(
    dataset_dir: str,
    informed_sender: bool,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    use_augmentations: bool = True,
    is_distributed: bool = False,
    seed: int = 111,
):
    transformations = ImageTransformation(image_size, use_augmentations)

    train_dataset = datasets.ImageFolder(dataset_dir, transform=transformations)
    train_sampler = None
    if is_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=True, drop_last=True, seed=seed
        )

    if informed_sender:
        collate_fn = collate_with_random_recv_input
    else:
        collate_fn = collate
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    return train_loader


class GaussianBlur:
    """Gaussian blur augmentation as in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class ImageTransformation:
    """
    A stochastic data augmentation module that transforms any given data example
    randomly resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, size: int, augmentation: bool = False):
        if augmentation:
            s = 1
            color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
            transformations = [
                transforms.RandomResizedCrop(size=size),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                transforms.RandomHorizontalFlip(),  # with 0.5 probability
            ]
        else:
            transformations = [transforms.Resize(size=(size, size))]

        transformations.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.transform = transforms.Compose(transformations)

    def __call__(self, x):
        return self.transform(x), self.transform(x)
