# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torch.utils.data.distributed import DistributedSampler


class GaussianDataset:
    def __init__(self, image_size: int, max_objects: int, nb_samples: int):
        self.image_size = image_size
        self.max_objects = max_objects
        self.nb_samples = nb_samples

    def __len__(self):
        return self.nb_samples

    def __getitem__(self, index):
        tnsr = torch.rand(self.max_objects, 3, self.image_size, self.image_size)
        return tnsr, torch.ones(self.max_objects)


def get_gaussian_dataloader(
    batch_size: int = 32,
    image_size: int = 32,
    max_objects: int = 20,
    is_distributed: bool = False,
    seed: int = 111,
    **kwargs,
):
    dataset = GaussianDataset(image_size, max_objects, nb_samples=20_000)

    sampler = None
    if is_distributed:
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=True, seed=seed)

    def collate_fn(batch):
        inp_tnsr = torch.stack([elem[0] for elem in batch])
        labels = torch.stack([elem[1] for elem in batch])
        baseline = torch.Tensor([1 / inp_tnsr.shape[1]] * len(batch))
        mask = torch.zeros(len(batch), max_objects)
        return inp_tnsr, labels, inp_tnsr, {"baselines": baseline, "mask": mask}

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        collate_fn=collate_fn,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    return loader
