# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license fod in the
# LICENSE file in the root directory of this source tree.

import glob
import json
import os
import random
import xml.etree.ElementTree as ET
from itertools import product
from pathlib import Path
from typing import Callable, Optional
from PIL import Image

import torch
import torch.distributed as dist
from torchvision import transforms
from torchvision.datasets import VisionDataset
from torchvision.transforms.functional import crop, resize
from torch.utils.data.distributed import DistributedSampler

from egg.zoo.referential_language.utils.data_utils import collate, ImageTransformation


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


class VisualObjectsDataset(VisionDataset):
    def __init__(
        self,
        image_dir: str,
        transform: Optional[Callable] = None,
    ):
        super(VisualObjectsDataset, self).__init__(root=image_dir, transform=transform)

    def __len__(self):
        return len(self.samples)

    def _load_and_transform(self, img_path):
        image = pil_loader(img_path)
        if self.transform:
            image = self.transform(image)
        return image

    def __getitem__(self, index):
        raise NotImplementedError


class Flickr30kDataset(VisualObjectsDataset):
    def __init__(
        self,
        image_dir: str = "/private/home/rdessi/flickr30k/",
        metadata_dir: str = "/private/home/rdessi/flickr30k/",
        split: str = "train",
        transform: Optional[Callable] = None,
        max_objects: int = 8,
        image_size: int = 64,
    ):
        super(Flickr30kDataset, self).__init__(image_dir=image_dir, transform=transform)
        metadata_dir = Path(metadata_dir)
        with open(metadata_dir / f"{split}.txt") as f:
            split_images = set(f.read().splitlines())

        ann_paths = glob.iglob(f"{os.path.expanduser(metadata_dir)}/Annotations/*xml")
        self.samples = []
        for ann_path in ann_paths:
            image_id = Path(ann_path).stem
            if image_id not in split_images:
                continue

            anns = get_annotations(ann_path)

            boxes = []
            for label, objs in anns["boxes"].items():
                boxes.extend(product([label], objs))
            if len(boxes) < 3:
                continue
            random.shuffle(boxes)
            anns["boxes"] = boxes

            img_path = Path(image_dir) / "Images" / f"{image_id}.jpg"
            sents = get_sentence_data(metadata_dir / "Sentences" / f"{image_id}.txt")

            self.samples.append((img_path, anns, sents))

            if len(self.samples) >= len(split_images):
                break

        self.transform = transform
        self.max_objects = max_objects
        self.resizer = transforms.Resize(size=(image_size, image_size))

    def __getitem__(self, index):
        img_path, anns, sents = self.samples[index]

        sender_image = self._load_and_transform(img_path)
        recv_image = self._load_and_transform(img_path)

        sender_objs, labels, recv_objs = [], [], []
        bboxes = []
        for label, coords in anns["boxes"][: min(self.max_objects, len(anns["boxes"]))]:
            xmin, ymin, xmax, ymax = coords
            x, y, w, h = xmin, ymin, xmax - xmin, ymax - ymin
            bboxes.append(torch.IntTensor([x, y, w, h]))

            sender_obj = self.resizer(crop(sender_image, y, x, h, w))
            recv_obj = self.resizer(crop(recv_image, y, x, h, w))

            sender_objs.append(sender_obj)
            recv_objs.append(recv_obj)

        sender_input = torch.stack(sender_objs)
        recv_input = torch.stack(recv_objs)
        labels = torch.Tensor(labels)
        aux = {
            "sender_image": resize(sender_image, size=(128, 128)),
            "recv_image": resize(recv_image, size=(128, 128)),
            "image_ids": torch.Tensor([int(img_path.stem)]),
            "image_sizes": torch.Tensor([*sender_image.shape]).int(),
            "bboxes": torch.stack(bboxes),
            "sents": sents,
        }

        return sender_input, labels, recv_input, aux


class VisualGenomeDataset(VisualObjectsDataset):
    def __init__(
        self,
        image_dir: str,
        metadata_dir: str,
        classes_path: str = "/private/home/rdessi/EGG/egg/zoo/referential_language/utils/classes_1600.txt",
        split: str = "train",
        transform: Callable = None,
        max_objects=10,
        image_size=64,
    ):
        super(VisualGenomeDataset, self).__init__(
            image_dir=image_dir, transform=transform
        )
        assert max_objects >= 3
        path_images = Path(image_dir)
        path_metadata = Path(metadata_dir) / f"{split}_objects.json"
        path_image_data = Path(metadata_dir) / f"{split}_image_data.json"

        with open(path_image_data) as img_in, open(path_metadata) as metadata_in:
            img_data, object_data = json.load(img_in), json.load(metadata_in)
        assert len(img_data) == len(object_data)

        self.class2id = {}
        idx = 0
        with open(classes_path) as f:
            for line in f:
                names = line.strip().split(",")
                for name in names:
                    self.class2id[name] = idx
                    idx += 1

        object_dict = {}
        for object_item in object_data:
            object_dict[object_item["image_id"]] = object_item

        self.samples = []
        for img_item in img_data:
            img_id = img_item["image_id"]
            object_item = object_dict[img_id]

            img_path = path_images / "/".join(img_item["url"].split("/")[-2:])

            self.samples.append((img_path, img_id, object_item["objects"]))

        self.id2class = {v: k for k, v in self.class2id.items()}
        self.transform = transform
        self.max_objects = max_objects
        self.resizer = transforms.Resize(size=(image_size, image_size))

    def __getitem__(self, index):
        img_path, img_id, obj_list = self.samples[index]

        sender_image = self._load_and_transform(img_path)
        recv_image = self._load_and_transform(img_path)

        sender_objs, labels, recv_objs = [], [], []
        bboxes = []
        for obj_item in obj_list[: min(self.max_objects, len(obj_list))]:
            x, y, w, h = obj_item["x"], obj_item["y"], obj_item["w"], obj_item["h"]
            bboxes.append(torch.IntTensor([x, y, w, h]))

            sender_obj = self.resizer(crop(sender_image, y, x, h, w))
            recv_obj = self.resizer(crop(recv_image, y, x, h, w))

            sender_objs.append(sender_obj)
            recv_objs.append(recv_obj)

            label = next(filter(lambda n: n in self.class2id, obj_item["names"]), None)
            assert label is not None
            labels.append(self.class2id[label])

        sender_input = torch.stack(sender_objs)
        recv_input = torch.stack(recv_objs)
        labels = torch.Tensor(labels)
        aux = {
            "sender_image": resize(sender_image, size=(128, 128)),
            "recv_image": resize(recv_image, size=(128, 128)),
            "image_ids": torch.Tensor([img_id]),
            "image_sizes": torch.Tensor([*sender_image.shape]).int(),
            "bboxes": torch.stack(bboxes),
        }

        return sender_input, labels, recv_input, aux


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
        aux_input = {"mask": torch.ones(self.max_objects).bool()}
        return x, labels, x, aux_input


def get_gaussian_dataloader(batch_size, image_size, max_objects, seed, **kwargs):
    ds = GaussianDataset(image_size, max_objects, nb_samples=1_000)

    sampler = None
    if dist.is_initialized():
        sampler = DistributedSampler(ds, shuffle=False, drop_last=True, seed=seed)

    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=6,
        sampler=sampler,
        pin_memory=True,
        drop_last=True,
    )


def get_dataloader(
    dataset_name: str,
    image_dir: str,
    metadata_dir: str,
    batch_size: int = 32,
    split: str = "train",
    image_size: int = 32,
    max_objects: int = 8,
    use_augmentation: bool = False,
    seed: int = 111,
):
    # image_dir: str = "/private/home/rdessi/flickr30k/Images",
    # metadata_dir: str = "/private/home/rdessi/flickr30k/Annotations",

    # image_dir: str = "/private/home/rdessi/visual_genome",
    # metadata_dir: str = "/private/home/rdessi/visual_genome/filtered_splits",

    assert dataset_name in ["flickr", "vg", "gaussian"]
    if dataset_name == "gaussian":
        return get_gaussian_dataloader(batch_size, image_size, max_objects, seed)

    name2dataset = {"flickr": Flickr30kDataset, "vg": VisualGenomeDataset}

    ds = name2dataset[dataset_name](
        image_dir=image_dir,
        metadata_dir=metadata_dir,
        split=split,
        transform=ImageTransformation(use_augmentation),
        max_objects=max_objects,
        image_size=image_size,
    )

    sampler = None
    if dist.is_initialized():
        sampler = DistributedSampler(
            ds, shuffle=(split != "test"), drop_last=True, seed=seed
        )

    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=8,
        sampler=sampler,
        collate_fn=collate,
        shuffle=(sampler is None and split != "test"),
        pin_memory=True,
        drop_last=True,
    )
