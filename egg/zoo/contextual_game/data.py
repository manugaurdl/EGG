# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from pathlib import Path
from PIL import Image

import torch
from torchvision import transforms


def img_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class ImageCodeDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, metadata_dir, split, transform):
        assert split in ["train", "valid"]

        self.image_dir = image_dir
        with open(Path(metadata_dir) / f"{split}_data.json", "r") as fd:
            data = json.load(fd)

        self.samples = []
        for img_dir, data in data.items():
            for img_idx, text in data.items():
                self.samples.append((img_dir, int(img_idx), text))

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_dir, img_idx, text = self.samples[idx]

        img_files = list((Path(self.image_dir) / img_dir).glob("*.jpg"))
        img_files.sort(key=lambda x: int(str(x).split("/")[-1].split(".")[0][3:]))

        images = [img_loader(photo_file) for photo_file in img_files]
        images = torch.stack([self.transform(photo) for photo in images])

        ground_truth = torch.tensor([img_idx]).long()

        return (
            images,
            ground_truth,
            images,
            {
                "captions": text,
                "is_video": "open-images" in img_dir,
                "input_images": images,
            },
        )


def collate(batch):
    elem = batch[0]
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.cat(batch, 0, out=out)
    elif isinstance(elem, bool) or isinstance(elem, str):
        return batch
    elif isinstance(elem, dict):
        return {key: collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, list) or isinstance(elem, tuple):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]

    raise RuntimeError("Cannot collate batch")


def get_dataloader(
    image_dir: str,
    metadata_dir: str,
    batch_size: int = 32,
    image_size: int = 32,
    num_workers: int = 0,
    split: str = "train",
    is_distributed: bool = False,
    seed: int = 111,
):
    transformations = [
        transforms.Resize(size=(image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    transformations = transforms.Compose(transformations)

    dataset = ImageCodeDataset(image_dir, metadata_dir, split, transformations)

    sampler = None
    if is_distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=(split != "valid"), drop_last=True, seed=seed
        )

    # Not using batch size due to nature of the task
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=sampler is None and split != "valid",
        sampler=sampler,
        collate_fn=collate,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )

    return loader
