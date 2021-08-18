# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from itertools import chain
from pathlib import Path
from typing import Callable, Optional

import torch
import torchvision
from bidict import bidict
from torchvision import VisionDataset, transforms
from PIL import Image, ImageFilter

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


def pop_last_target(targets):
    return targets.pop()  # remove and return last element of list


class OpenImageDataset(VisionDataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        split: str = "validation",
        choose_target_fn: Optional[Callable] = pop_last_target,
    ):
        super(OpenImageDataset, self).__init__(
            root, transform=transform, target_transform=target_transform
        )
        images_folder = root / split / "images"
        if split == "train":
            all_folders = images_folder.glob(r"train_0[0-9]")
            all_images = chain(*[folder.glob(r"*.jpg") for folder in all_folders])
        else:
            all_images = (images_folder / split).glob(r"*.jpg")

        bbox_csv_filepath = root.joinpath(split, f"{split}-annotations-bbox.csv")
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
            root.joinpath("metadata", "class-descriptions-boxable.csv"),
            discard_header=False,
        )
        self.label_name_to_id = bidict(
            zip(
                self.label_name_to_class_description.keys(),
                range(len(self.label_name_to_class_description.keys())),
            )
        )

        self.transform = transform
        self.target_transform = target_transform

        self.choose_target_fn = choose_target_fn

    def __len__(self) -> int:
        return len(self.images)

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

        bbox[0] *= image.shape[1]  # assuming image is a tensor of size C x H x W
        bbox[1] *= image.shape[1]
        bbox[2] *= image.shape[0]
        bbox[3] *= image.shape[0]

        # Xmin, Xmax, Ymin, Ymax
        bbox = torch.stack(torch.Tensor([bbox[0], bbox[2], bbox[1], bbox[3]]))
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


class GaussianBlur:
    """Gaussian blur augmentation as in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class ImageTransformation:
    def __init__(
        self,
        size: int,
        augmentation: bool = False,
    ):
        if augmentation:
            s = 1
            color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
            transformations = [
                transforms.RandomResizedCrop(size=size),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                transforms.RandomHorizontalFlip(),  # with 0.5 probability
            ]
        else:
            transformations = [transforms.Resize(size=(size, size))]

        transformations.extend([transforms.ToTensor()])

        self.transform = transforms.Compose(transformations)

    def __call__(self, x):
        return self.transform(x), self.transform(x)


class BoxResize:
    def __init__(self, new_size: int, height_first: bool = True):
        self.new_size = new_size
        self.height_first = height_first

    def __call__(self, boxes, original_size):
        ratios = [s / s_orig for s, s_orig in zip(self.new_size, original_size)]

        if self.height_first:
            ratio_height, ratio_width = ratios
        else:
            ratio_width, ratio_height = ratios
        xmin, ymin, xmax, ymax = boxes

        xmin = xmin * ratio_width
        xmax = xmax * ratio_width
        ymin = ymin * ratio_height
        ymax = ymax * ratio_height
        new_coords = torch.Tensor((xmin, ymin, xmax, ymax))
        return new_coords


if __name__ == "__main__":
    transform = ImageTransformation(size=224, augmentation=False)
    a = OpenImageDataset(
        Path("/datasets01/open_images/030119"),
        split="validation",
        transform=transform,
        target_transform=BoxResize(224),
    )
    for idx, i in enumerate(a):
        if idx == 20:
            break
        continue
