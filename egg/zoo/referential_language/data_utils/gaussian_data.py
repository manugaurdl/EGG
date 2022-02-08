# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license fod in the
# LICENSE file in the root directory of this source tree.

import torch


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

        game_labels = torch.arange(self.max_objects)
        mask = torch.ones(self.max_objects, dtype=torch.bool)
        baseline = torch.Tensor([1 / self.max_objects])
        aux_input = {"mask": mask, "game_labels": game_labels, "baseline": baseline}

        return x, labels, torch.Tensor(), aux_input


def get_gaussian_dataloader(image_size, max_objects, batch_size, **kwargs):
    return torch.utils.data.DataLoader(
        GaussianDataset(image_size, max_objects, nb_samples=2_000),
        batch_size=batch_size,
        num_workers=12,
        pin_memory=True,
        drop_last=True,
    )
