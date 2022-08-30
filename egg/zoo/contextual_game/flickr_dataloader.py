# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license fod in the
# LICENSE file in the root directory of this source tree.

import glob
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Optional
from PIL import Image

import torch
from torch.nn.functional import pad
from torchvision import transforms
from torchvision.datasets import VisionDataset
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


def get_sentence_data(fn):
    """
    Parses a sentence file from the Flickr30K Entities dataset

    input:
      fn - full file path to the sentence file to parse

    output:
      a list of dictionaries for each sentence with the following fields:
          sentence - the original sentence
          phrases - a list of dictionaries for each phrase with the
                    following fields:
                      phrase - the text of the annotated phrase
                      first_word_index - the position of the first word of
                                         the phrase in the sentence
                      phrase_id - an identifier for this phrase
                      phrase_type - a list of the coarse categories this
                                    phrase belongs to

    """
    with open(fn, "r") as f:
        sentences = f.read().split("\n")

    annotations = []
    for sentence in sentences:
        if not sentence:
            continue

        first_word = []
        phrases = []
        phrase_id = []
        phrase_type = []
        words = []
        current_phrase = []
        add_to_phrase = False
        for token in sentence.split():
            if add_to_phrase:
                if token[-1] == "]":
                    add_to_phrase = False
                    token = token[:-1]
                    current_phrase.append(token)
                    phrases.append(" ".join(current_phrase))
                    current_phrase = []
                else:
                    current_phrase.append(token)

                words.append(token)
            else:
                if token[0] == "[":
                    add_to_phrase = True
                    first_word.append(len(words))
                    parts = token.split("/")
                    phrase_id.append(parts[1][3:])
                    phrase_type.append(parts[2:])
                else:
                    words.append(token)

        sentence_data = {"sentence": " ".join(words), "phrases": []}
        for index, phrase, p_id, p_type in zip(
            first_word, phrases, phrase_id, phrase_type
        ):
            sentence_data["phrases"].append(
                {
                    "first_word_index": index,
                    "phrase": phrase,
                    "phrase_id": p_id,
                    "phrase_type": p_type,
                }
            )

        annotations.append(sentence_data)

    return annotations


def get_annotations(fn):
    """
    Parses the xml files in the Flickr30K Entities dataset

    input:
      fn - full file path to the annotations file to parse

    output:
      dictionary with the following fields:
          scene - list of identifiers which were annotated as
                  pertaining to the whole scene
          nobox - list of identifiers which were annotated as
                  not being visible in the image
          boxes - a dictionary where the fields are identifiers
                  and the values are its list of boxes in the
                  [xmin ymin xmax ymax] format
    """
    tree = ET.parse(fn)
    root = tree.getroot()
    size_container = root.findall("size")[0]
    anno_info = {"boxes": {}, "scene": [], "nobox": []}
    for size_element in size_container:
        anno_info[size_element.tag] = int(size_element.text)

    for object_container in root.findall("object"):
        for names in object_container.findall("name"):
            box_id = names.text
            box_container = object_container.findall("bndbox")
            if len(box_container) > 0:
                if box_id not in anno_info["boxes"]:
                    anno_info["boxes"][box_id] = []
                xmin = int(box_container[0].findall("xmin")[0].text) - 1
                ymin = int(box_container[0].findall("ymin")[0].text) - 1
                xmax = int(box_container[0].findall("xmax")[0].text) - 1
                ymax = int(box_container[0].findall("ymax")[0].text) - 1
                anno_info["boxes"][box_id].append([xmin, ymin, xmax, ymax])
            else:
                nobndbox = int(object_container.findall("nobndbox")[0].text)
                if nobndbox > 0:
                    anno_info["nobox"].append(box_id)

                scene = int(object_container.findall("scene")[0].text)
                if scene > 0:
                    anno_info["scene"].append(box_id)

    return anno_info


def pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class Flickr30kDataset(VisionDataset):
    def __init__(
        self,
        dataset_dir: str = "/checkpoint/rdessi/datasets/flickr30k",
        split: str = "train",
        transform: Optional[Callable] = None,
    ):
        super(Flickr30kDataset, self).__init__(dataset_dir, transform=transform)
        dataset_dir = Path(dataset_dir)
        with open(dataset_dir / f"{split}.txt") as f:
            split_images = set(f.read().splitlines())

        ann_paths = glob.iglob(f"{os.path.expanduser(dataset_dir)}/Annotations/*xml")
        self.samples = []
        for ann_path in ann_paths:
            image_id = Path(ann_path).stem
            if image_id not in split_images:
                continue

            img_path = Path(dataset_dir) / "Images" / f"{image_id}.jpg"
            anns = get_annotations(ann_path)
            sents = get_sentence_data(dataset_dir / "Sentences" / f"{image_id}.txt")

            self.samples.append((img_path, anns, sents))

            if len(self.samples) >= len(split_images):
                break

        self.transform = transform
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, anns, sents = self.samples[index]

        target = pil_loader(img_path)
        if self.transform:
            target = self.transform(target)

        candidates = [
            self.transform(pil_loader(self.samples[idx][0]))
            for idx in random.choices(range(len(self.samples)), k=9)
        ]
        candidates.append(target)

        label = torch.Tensor([random.randint(0, 9)]).long()

        candidates[label], candidates[-1] = candidates[-1], candidates[label]
        candidates = torch.stack(candidates, dim=0)

        captions = [elem["sentence"] for elem in sents]
        tokenized_text = self.tokenizer(
            [captions[0]],
            max_length=200,
            truncation=True,
            return_tensors="pt",
        )
        caption_len = tokenized_text["input_ids"].shape[1]

        aux = {
            "text": captions[0],
            "captions": pad(
                tokenized_text["input_ids"],
                pad=(0, max(0, 200 - caption_len)),
                value=-1,
            ),
        }
        return target.unsqueeze(0), label, candidates, aux


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


def get_dataloader(
    dataset_dir: str = "/checkpoint/rdessi/datasets/flickr30k",
    batch_size: int = 32,
    image_size: int = 32,
    split: str = "train",
    num_workers: int = 8,
):

    ds = Flickr30kDataset(
        dataset_dir=dataset_dir, split=split, transform=get_transform(image_size)
    )

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=split != "test",
        collate_fn=collate,
        pin_memory=True,
        drop_last=True,
    )
    return loader


if __name__ == "__main__":
    dataset_dir = "/checkpoint/rdessi/datasets/flickr30k"
    dl = get_dataloader(
        dataset_dir=dataset_dir,
        split="test",
        batch_size=8,
        num_workers=0,
    )
