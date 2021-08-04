# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from itertools import chain
from pathlib import Path
from typing import Callable, List

import torch
import torchvision
from bidict import bidict
from PIL import Image
from torch.utils.data import Dataset

from egg.zoo.referential_language.utils_data import csv_to_dict, multicolumn_csv_to_dict


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


def random_target(labels: List):
    return labels[0]


class OpenImagesObjects(Dataset):
    def __init__(
        self,
        root_folder: Path,
        split: str = "validation",
        choose_target_fn: Callable = random_target,
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

        self.choose_target_fn = choose_target_fn
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def resize_bbox(self, boxes, original_size, new_size):
        ratios = []
        for s, s_orig in zip(new_size, original_size):
            ratios.append(
                torch.tensor(s, dtype=torch.float32, device=boxes.device)
                / torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
            )

        ratio_height, ratio_width = ratios
        xmin, ymin, xmax, ymax = boxes

        xmin = xmin * ratio_width
        xmax = xmax * ratio_width
        ymin = ymin * ratio_height
        ymax = ymax * ratio_height
        new_coords = torch.Tensor((xmin, ymin, xmax, ymax))
        return new_coords

    def __getitem__(self, index: int):
        image_path = self.images[index]
        image = Image.open(image_path)

        if image.mode != "RGB":
            image.convert("RGB")
        if self.transform:
            image = self.transform(image)  # NOTE PIL image is in W, H format

        labels = self.box_labels[image_path.stem]
        label = self.choose_target_fn(labels)
        obj_label, *bbox = label

        bbox = torch.tensor([float(coord) for coord in bbox])
        bbox[0] *= image.size[0]
        bbox[1] *= image.size[0]
        bbox[2] *= image.size[1]
        bbox[3] *= image.size[1]

        # Xmin, Xmax, Ymin, Ymax
        bbox = torch.stack([bbox[0], bbox[2], bbox[1], bbox[3]])
        img = torchvision.transforms.functional.pil_to_tensor(image)

        cropped_target = torchvision.transforms.functional.crop(
            img,
            top=round(bbox[0].item()),
            left=round(bbox[2].item()),
            height=round(bbox[2].item() + bbox[3].item()),
            width=round(bbox[0].item() + bbox[1].item()),
        )

        img = torchvision.transforms.functional.resize(img, size=(224, 224))
        cropped_target = torchvision.transforms.functional.resize(
            cropped_target, size=img.shape[-2:]
        )
        bbox = self.resize_bbox(bbox, original_size=img.shape, new_size=(224, 224))
        label = self.label_name_to_class_description[obj_label]

        torch.save(
            (img, cropped_target, label, bbox),
            f"/private/home/rdessi/random_samples_img/prova_{index}.image",
        )

        # bbox_area = (bbox[1] - bbox[0]) * (bbox[3] - bbox[2])
        # perc_area = bbox_area / image.shape[2] * image.shape[1]
        obj_id = torch.tensor(self.label_name_to_id[obj_label])  # model label
        return image, label, obj_id


if __name__ == "__main__":
    a = OpenImagesObjects(Path("/datasets01/open_images/030119"), split="validation")
    for idx, i in enumerate(a):
        if idx == 20:
            break
        continue
