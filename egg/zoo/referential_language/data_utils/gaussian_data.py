# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch


class GaussianDataset:
    def __init__(self, image_size: int, nb_samples: int):
        self.image_size = image_size
        self.nb_samples = nb_samples

    def __len__(self):
        return self.nb_samples

    def __getitem__(self, index):
        x = torch.rand(3, self.image_size, self.image_size)
        labels = torch.ones(1)
        return x, labels, x


def get_gaussian_dataloader(image_size, max_objects, **kwargs):
    return torch.utils.data.DataLoader(
        GaussianDataset(image_size, nb_samples=2_000),
        batch_size=max_objects,
        num_workers=6,
        pin_memory=True,
        drop_last=True,
    )
