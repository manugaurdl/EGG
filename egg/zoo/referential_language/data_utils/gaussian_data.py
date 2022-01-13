# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json

import torch

from egg.core.batch import Batch


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


def gaussian_eval(game, data, device):
    n_batches = 0
    acc = 0.0
    game.eval()
    with torch.no_grad():
        for batch_id, batch in enumerate(data):
            if not isinstance(batch, Batch):
                batch = Batch(*batch)
            batch = batch.to(device)
            _, interaction = game(*batch)
            acc += interaction.aux["acc"].mean().item()
            n_batches += 1

    dump = {
        "acc": acc / n_batches,
        "baseline": 1 / interaction.aux["acc"].shape[0],
        "mode": "GAUSSIAN TEST",
    }
    print(json.dumps(dump), flush=True)


def get_gaussian_dataloader(image_size, max_objects, **kwargs):
    return torch.utils.data.DataLoader(
        GaussianDataset(image_size, nb_samples=2_000),
        batch_size=max_objects,
        num_workers=6,
        pin_memory=True,
        drop_last=True,
    )
