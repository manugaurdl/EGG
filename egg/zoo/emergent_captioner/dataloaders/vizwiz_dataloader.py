# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from pathlib import Path
from typing import Callable
from PIL import Image

import torch
import torch.distributed as dist


from egg.zoo.emergent_captioner.dataloaders.utils import MyDistributedSampler


class VizWizDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dir,
        split,
        transform,
    ):
        assert split in ["train", "valid", "test"]

        self.dataset_dir = dataset_dir
        with open(dataset_dir / f"{split}_data.json", "r") as fd:
            data = json.load(fd)

        breakpoint()
        self.samples = []
        for img_dir, sents in data.items():
            for img_idx, text in sents.items():
                self.samples.append((img_dir, int(img_idx), text))

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        set_dir, img_idx, text = self.samples[idx]
        set_path = Path(self.dataset_dir) / "image-sets" / set_dir
        img_files = list(set_path.glob("*.jpg"))
        img_files.sort(key=lambda x: int(str(x).split("/")[-1].split(".")[0][3:]))

        images = [Image.open(photo_file) for photo_file in img_files]
        images[0], images[img_idx] = images[img_idx], images[0]

        sender_imgs, recv_imgs = zip(*[self.transform(photo) for photo in images])

        sender_input = torch.stack(sender_imgs)[torch.LongTensor([0])]
        recv_input = torch.stack(recv_imgs)

        aux_input = {
            "captions": text,
            "is_video": torch.Tensor(["open-images" not in set_dir]),
            "target_idx_imagecode": torch.tensor([img_idx]).long(),
        }
        return sender_input, torch.tensor([idx]), recv_input, aux_input


class VizWizWrapper:
    def __init__(self, dataset_dir: str = None):
        if dataset_dir is None:
            dataset_dir = "/checkpoint/rdessi/datasets/vizwiz"
        self.dataset_dir = Path(dataset_dir)

    def get_split(
        self,
        split: str,
        transform: Callable,
        batch_size: int = 1,
        num_workers: int = 8,
        shuffle: bool = None,
        seed: int = 111,
    ):
        ds = VizWizDataset(self.dataset_dir, split=split, transform=transform)

        sampler = None
        if dist.is_initialized():
            if shuffle is None:
                shuffle = split != "test"
            sampler = MyDistributedSampler(
                ds, shuffle=shuffle, drop_last=True, seed=seed
            )

        if shuffle is None:
            shuffle = split != "test" and sampler is None

        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=sampler is None and split != "test",
            sampler=sampler,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
        )

        return loader


if __name__ == "__main__":
    w = VizWizWrapper()
    dl = w.get_split(split="test", num_workers=0)
    for i, elem in enumerate(dl):
        if i == 10:
            break
        continue
