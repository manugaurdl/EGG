# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

import torch


def get_gaussian_noise_dataloader(
    dataset_size: int = 49152,
    batch_size: int = 128,
    image_size: int = 224,
    num_workers: int = 4,
    **kwargs
):
    dataset = GaussianNoiseDataset(size=dataset_size, image_size=image_size)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader


class GaussianNoiseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_size: int,
        image_size: int = 64,
        transformations: Optional[Callable] = None,
    ):
        self.image_size = image_size
        self.dataset_size = dataset_size

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        sample = torch.randn(3, self.image_size, self.image_size)
        return sample, torch.zeros(1), sample
