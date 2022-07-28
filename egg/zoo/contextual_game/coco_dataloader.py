# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from PIL import Image

import torch
from torch.nn.functional import pad
from torchvision import transforms
from torchvision.datasets import CocoCaptions
from transformers import GPT2Tokenizer

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def get_transform(image_size: int):
    def _convert_image_to_rgb(image: Image.Image):
        return image.convert("RGB")

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
    return transforms.Compose(transformations)


class CocoDataset(CocoCaptions):
    def __init__(
        self,
        root,
        annFile,
        transform=None,
        n_distractors=9,
    ):
        super(CocoDataset, self).__init__(root, annFile, transform)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.n_distractors = n_distractors

        random.shuffle(self.ids)

        self.ids = self.ids[:2300]

    def __getitem__(self, idx):
        target, captions = super().__getitem__(idx)

        candidates = [
            super(type(self), self).__getitem__(idx)[0]
            for idx in random.choices(range(len(self.ids)), k=self.n_distractors)
        ]
        candidates.append(target)

        tokenized_text = self.tokenizer(
            [captions[0]],
            max_length=200,
            truncation=True,
            return_tensors="pt",
        )

        label = torch.Tensor([random.randint(0, self.n_distractors)]).long()

        candidates[label], candidates[-1] = candidates[-1], candidates[label]
        candidates = torch.stack(candidates, dim=0)

        caption_len = tokenized_text["input_ids"].shape[1]

        return (
            target.unsqueeze(0),
            label,
            candidates,
            {
                "captions": pad(
                    tokenized_text["input_ids"],
                    pad=(0, max(0, 200 - caption_len)),
                    value=-1,
                ),
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
    image_size: int = 224,
    num_workers: int = 8,
):
    ds = CocoDataset(
        root=image_dir, annFile=metadata_dir, transform=get_transform(image_size)
    )

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        drop_last=True,
        pin_memory=True,
    )
    return loader


if __name__ == "__main__":
    image_dir = "/datasets01/COCO/060817/val2014"
    metadata_dir = "/datasets01/COCO/060817/annotations/captions_val2014.json"
    dl = get_dataloader(
        image_dir=image_dir,
        metadata_dir=metadata_dir,
        batch_size=8,
        num_workers=0,
    )
