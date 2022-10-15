# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from pathlib import Path
from PIL import Image

import torch
import torch.distributed as dist
from torchvision import transforms

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from egg.zoo.emergent_captioner.dataloaders.utils import MyDistributedSampler


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


def _convert_image_to_rgb(image: Image.Image):
    return image.convert("RGB")


class ImageCodeDataset(torch.utils.data.Dataset):
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
        images = torch.stack([self.transform(photo) for photo in images])

        aux_input = {
            "caption": text,
            "is_video": torch.Tensor(["open-images" not in set_dir]),
            "target_idx": torch.tensor([img_idx]).long(),
        }
        return images[torch.LongTensor([0])], torch.tensor([idx]), images, aux_input


class ImageCodeWrapper:
    def __init__(self, dataset_dir: str = None):
        if dataset_dir is None:
            dataset_dir = "/checkpoint/rdessi/datasets/imagecode"
        self.dataset_dir = Path(dataset_dir)

    def get_split(
        self,
        split: str,
        batch_size: int = 1,
        image_size: int = 224,
        num_workers: int = 8,
        shuffle: bool = None,
        seed: int = 111,
    ):
        transformations = [
            transforms.Resize(image_size, interpolation=BICUBIC),
            transforms.CenterCrop(image_size),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
        transformations = transforms.Compose(transformations)
        ds = ImageCodeDataset(self.dataset_dir, split=split, transform=transformations)

        sampler = None
        if dist.is_initialized():
            if shuffle is None:
                shuffle = split != "test"
            sampler = MyDistributedSampler(
                ds, shuffle=shuffle, drop_last=True, seed=seed
            )

        if shuffle is None:
            shuffle = split != "test" and sampler is None

        # Setting batch to 1 since batching is handled by the update_freq EGG parameter
        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=1,
            shuffle=sampler is None and split != "test",
            sampler=sampler,
            collate_fn=collate,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
        )

        return loader


if __name__ == "__main__":
    w = ImageCodeWrapper()
    dl = w.get_split(split="test", num_workers=0)
    for i, elem in enumerate(dl):
        if i == 10:
            break
        continue
