# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import (
    DefaultDict,
    Dict,
    List,
    Iterable,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)
from collections import defaultdict
from itertools import chain
from operator import itemgetter, methodcaller
from pathlib import Path
from typing import Callable

import torch
from bidict import bidict
from PIL import Image
from torch.utils.data import Dataset


K = TypeVar("K")
V = TypeVar("V")

BBOX_INDICES = {
    "ImageID": 0,
    "Source": 1,
    "LabelName": 2,
    "Confidence": 3,
    "XMin": 4,
    "XMax": 5,
    "YMin": 6,
    "YMax": 7,
    "IsOccluded": 8,
    "IsTruncated": 9,
    "IsGroupOf": 10,
    "IsDepiction": 11,
    "IsInside": 12,
}


class OpenImagesObjects(Dataset):
    def __init__(
        self,
        root_folder: Path,
        split: str = "validation",
        transform: Callable = None,
    ):
        super().__init__()
        images_folder = root_folder / split / "images"
        if split == "train":
            all_folders = images_folder.glob(r"train_0[0-9]")
            all_images = chain(*[folder.glob(r"*.jpg") for folder in all_folders])
        else:
            all_images = (images_folder / split).glob(r"*.jpg")

        bbox_csv_filepath = root_folder.joinpath(split, f"{split}-annotations-bbox.csv")
        indices = tuple(
            BBOX_INDICES[key]
            for key in (
                "LabelName",
                "XMin",
                "XMax",
                "YMin",
                "YMax",
            )
        )
        self.box_labels = multicolumn_csv_to_dict(
            bbox_csv_filepath, value_cols=indices, one_to_n_mapping=True
        )

        images_with_labels = set(self.box_labels.keys())
        self.images = [
            image_path
            for image_path in all_images
            if image_path.stem in images_with_labels
        ]

        self.label_name_to_class_description = csv_to_dict(
            root_folder.joinpath("metadata", "class-descriptions-boxable.csv"),
            discard_header=False,
        )
        self.label_name_to_id = bidict(
            zip(
                self.label_name_to_class_description.keys(),
                range(len(self.label_name_to_class_description.keys())),
            )
        )

        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def prep_labels(self, labels, image):
        obj_label, *bbox = labels

        bbox = torch.tensor(list(map(float, bbox)))
        bbox[0] *= image.shape[1]
        bbox[1] *= image.shape[0]
        bbox[2] *= image.shape[1]
        bbox[3] *= image.shape[0]

        #  torchvision.transforms.functional.crop(image, top, left, height, width)

        obj_id = torch.tensor(self.label_name_to_id[obj_label])

        return (obj_id, bbox)

    def __getitem__(self, index: int):
        image_path = self.images[index]
        image = Image.open(image_path)

        if image.mode != "RGB":
            image.convert("RGB")
        if self.transform:
            image = self.transform(image)

        labels = self.box_labels[image_path.stem]
        labels = [self.prep_labels(label, image.size) for label in labels]
        return image, labels


def default_dict(pairs: Iterable[Tuple[K, V]]) -> DefaultDict[K, V]:
    mapping = defaultdict(list)
    for key, val in pairs:
        mapping[key].append(val)
    return mapping


def read_csv(file_path: Path, discard_header: bool = True) -> List[List[str]]:
    with open(file_path) as text_file:
        text = text_file.read()

    lines = text.split("\n")
    _ = lines.pop() if lines[-1] == "" else None  # pop final empty line if present
    print(file_path, len(lines))
    # [x.split(',') for x in lines]
    table = map(methodcaller("split", ","), lines)
    if discard_header:
        next(table)
    return list(table)


def csv_to_dict(
    file_path: Path,
    key_col: int = 0,
    value_col: int = 1,
    discard_header: bool = True,
    one_to_n_mapping: bool = False,
) -> Union[Dict[str, str], DefaultDict[str, str]]:
    table = read_csv(file_path, discard_header)
    # ((line[key_col], line[value_col]) for line in table)
    pairs = map(itemgetter(key_col, value_col), table)
    if one_to_n_mapping:
        return default_dict(pairs)
    return dict(pairs)


def multicolumn_csv_to_dict(
    file_path: Path,
    key_cols: Sequence = (0,),
    value_cols: Optional[Sequence] = None,
    discard_header: bool = True,
    one_to_n_mapping: bool = False,
):
    table = read_csv(file_path, discard_header)
    if not value_cols:
        value_cols = tuple(i for i in range(1, len(table[0])))
    # (tuple(line[i] for i in key_cols) for line in table)
    key_columns = map(itemgetter(*key_cols), table)
    value_columns = map(itemgetter(*value_cols), table)
    pairs = zip(key_columns, value_columns)
    if one_to_n_mapping:
        return default_dict(pairs)
    return dict(pairs)


if __name__ == "__main__":
    a = OpenImagesObjects(Path("/datasets01/open_images/030119"), split="validation")
    for i in a:
        continue
    print(a.lab / len(a))
    breakpoint()
