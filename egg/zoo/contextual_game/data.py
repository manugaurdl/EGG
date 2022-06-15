# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from pathlib import Path
from PIL import Image

import torch
from torch.nn.functional import pad
from torchvision import transforms
from transformers import GPT2Tokenizer

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


class ImageCodeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_dir,
        metadata_dir,
        split,
        transform,
        max_caption_len=150,
    ):
        assert split in ["train", "valid", "test"]

        self.image_dir = image_dir
        with open(Path(metadata_dir) / f"{split}_data.json", "r") as fd:
            data = json.load(fd)

        self.samples = []
        for img_dir, sents in data.items():
            for img_idx, text in sents.items():
                self.samples.append((img_dir, int(img_idx), text))

        self.transform = transform

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        print(f"| INFO: dataloader uses {type(self.tokenizer).__name__}")

        self.max_caption_len = max_caption_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_dir, img_idx, text = self.samples[idx]
        img_files = list((Path(self.image_dir) / img_dir).glob("*.jpg"))
        img_files.sort(key=lambda x: int(str(x).split("/")[-1].split(".")[0][3:]))

        images = [Image.open(photo_file) for photo_file in img_files]
        images = torch.stack([self.transform(photo) for photo in images])

        tokenized_text = self.tokenizer(
            [text],
            max_length=self.max_caption_len,
            truncation=True,
            return_tensors="pt",
        )
        caption_len = tokenized_text["input_ids"].shape[1]

        ground_truth = torch.tensor([img_idx]).long()

        return (
            images[ground_truth],
            ground_truth,
            images,
            {
                "captions": pad(
                    tokenized_text["input_ids"],
                    pad=(0, max(0, self.max_caption_len - caption_len)),
                    value=-1,
                ),
                "is_video": torch.Tensor(["open-images" not in img_dir]),
                # "input_images": images,
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


def _convert_image_to_rgb(image: Image.Image):
    return image.convert("RGB")


def get_dataloader(
    image_dir: str,
    metadata_dir: str,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 8,
    split: str = "train",
    is_distributed: bool = False,
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
    dataset = ImageCodeDataset(image_dir, metadata_dir, split, transformations)

    sampler = None
    if is_distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=(split != "test"), drop_last=True, seed=seed
        )

    # Setting batch to 1 since batching is handled by the update_freq EGG parameter
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=sampler is None and split != "test",
        sampler=sampler,
        collate_fn=collate,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )

    return loader
