# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List

import torch
from torchvision import datasets

from egg.zoo.emcom_as_ssl.utils_data import ImageTransformation


class CollaterWithRandomRecv:
    def __init__(self, batch_size: int, game_size: int):
        self.batch_size = batch_size
        self.game_size = game_size

    def __call__(self, batch: List[Any]):
        adjusted_bsz = self.batch_size // self.game_size
        sender_input, receiver_input, class_labels = [], [], []
        for elem in batch:
            sender_input.append(elem[0][0])
            receiver_input.append(elem[0][1])
            class_labels.append(torch.LongTensor([elem[1]]))

        sender_input = torch.stack(sender_input)
        receiver_input = torch.stack(receiver_input)
        class_labels = torch.stack(class_labels).view(
            adjusted_bsz, 1, self.game_size, -1
        )

        random_order = torch.stack(
            [torch.randperm(self.game_size) for _ in range(adjusted_bsz)]
        )
        target_position = torch.argmin(random_order, dim=1)

        return (
            sender_input,
            class_labels,
            receiver_input,
            {"target_position": target_position, "random_order": random_order},
        )


def get_dataloader(
    dataset_dir: str,
    image_size: int = 224,
    batch_size: int = 32,
    game_size: int = 2,
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

    collater = CollaterWithRandomRecv(batch_size=batch_size, game_size=game_size)
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
