# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import random
from PIL import ImageFilter
from typing import Any, List

import torch
from torchvision import datasets, transforms


class Collater:
    def __init__(
        self,
        batch_size=128,
        distractors=1,
        transformations=None,
    ):
        self.batch_size = batch_size
        assert (
            self.batch_size % 2 == 0
        ), f"batch_size my be a multiple of 2, found {batch_size} instead"
        self.distractors = distractors
        assert self.distractors == 1, "currently only one distractors is supported"
        self.transformations = transformations

    def __call__(self, batch: List[Any]):
        # this piece of code does not work with self.distractors > 1
        sender_input, receiver_input, class_labels = [], [], []
        for elem in batch:
            sender_input.append(elem[0][0])
            receiver_input.append(elem[0][1])
            class_labels.append(torch.LongTensor([elem[1]]))

        sender_input = torch.stack(sender_input)
        receiver_input = torch.stack(receiver_input)
        class_labels = torch.stack(class_labels)

        random_order = torch.randperm(len(batch))
        receiver_input = receiver_input[random_order]
        target_position = torch.argmin(random_order).unsqueeze(0)

        return sender_input, (class_labels, target_position), receiver_input


def get_dataloader(
    dataset_dir: str,
    image_size: int = 224,
    batch_size: int = 128,
    n_distractors: int = 1,
    num_workers: int = 4,
    use_augmentations: bool = True,
    is_distributed: bool = False,
    return_original_image: bool = False,
    seed: int = 111,
):
    print(f"using {n_distractors} distractors")
    transformations = ImageTransformation(image_size, use_augmentations)

    train_dataset = datasets.ImageFolder(dataset_dir, transform=transformations)
    train_sampler = None
    if is_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=True, drop_last=True, seed=seed
        )

    collater = Collater(
        batch_size=batch_size,
        distractors=n_distractors,
        transformations=transformations,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collater,
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
