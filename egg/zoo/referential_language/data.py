# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random

from itertools import chain
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Union

import torch
from bidict import bidict
from PIL import Image, ImageFilter
from torchvision import transforms
import torchvision.transforms.functional as tr_F
from torchvision.datasets.vision import VisionDataset

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


class Collater:
    def __init__(self, contextual_distractors: bool):
        if contextual_distractors:
            self.collate_fn = self._collate_with_contextual_distractors
        else:
            self.collate_fn = self._collate_with_random_distractors

    def _collate_with_random_distractors(self, batch: List[Any]) -> List[torch.Tensor]:
        sender_input, receiver_input, class_labels = [], [], []
        for elem in batch:
            sender_input.append(elem[0])
            class_labels.append(torch.LongTensor([elem[1]]))
            receiver_input.append(elem[2])

        sender_input = torch.stack(sender_input)
        receiver_input = torch.stack(receiver_input)
        class_labels = torch.stack(class_labels)

        return sender_input, class_labels, receiver_input

    def _collate_with_contextual_distractors(
        self, batch: List[Any]
    ) -> List[torch.Tensor]:
        # batch.sort(key=lambda x: x[-1], reverse=True)

        max_batch_size = len(batch)
        curr_batch_size = 0

        sender_input, labels, receiver_input, elem_idx_in_batch = [], [], [], []

        for idx, elem in enumerate(batch):
            if curr_batch_size + int(elem[3].item()) >= max_batch_size:
                missing_elems = max_batch_size - curr_batch_size

                elem = (
                    elem[0][:missing_elems],
                    elem[1][:missing_elems],
                    elem[2][:missing_elems],
                    torch.LongTensor([missing_elems]),
                )

            sender_input.append(elem[0])
            labels.append(elem[1])
            receiver_input.append(elem[2])
            elem_idx_in_batch.append(torch.LongTensor([idx for _ in range(elem[3])]))

            curr_batch_size += elem[3].item()

            if curr_batch_size == max_batch_size:
                break

        assert curr_batch_size == max_batch_size
        return (
            torch.cat(sender_input),
            torch.cat(labels),
            torch.cat(receiver_input),
            torch.cat(elem_idx_in_batch),
        )

    def __call__(self, batch: List[Any]) -> List[torch.Tensor]:
        return self.collate_fn(batch)


def get_dataloader(
    dataset_dir: str = "/datasets01/open_images/030119",
    batch_size: int = 128,
    num_workers: int = 4,
    contextual_distractors: bool = False,
    image_size: int = 64,
    use_augmentations: bool = True,
    is_distributed: bool = False,
    seed: int = 111,
):

    transform = ImageTransformation(size=image_size, augmentation=use_augmentations)

    train_dataset = OpenImages(
        Path(dataset_dir),
        split="train",
        transform=transform,
        contextual_distractors=contextual_distractors,
    )

    train_sampler = None
    if is_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=True, drop_last=True, seed=seed
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=Collater(contextual_distractors=contextual_distractors),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return train_loader


class OpenImages(VisionDataset):
    def __init__(
        self,
        root_folder: Union[Path, str] = "/datasets01/open_images/030119",
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        contextual_distractors: bool = False,
    ):
        super(OpenImages, self).__init__(
            root=root_folder, transform=transform, target_transform=target_transform
        )
        if isinstance(root_folder, str):
            root_folder = Path(root_folder)
        images_folder = root_folder / split / "images"
        if split == "train":
            all_folders = images_folder.glob(r"train_0[0-9]")
            all_images = chain(*[folder.glob(r"*.jpg") for folder in all_folders])
        else:
            all_images = (images_folder / split).glob(r"*.jpg")

        if split == "train":
            bbox_csv_filepath = (
                "/private/home/rdessi/contextual_emcomm/train-annotations-bbox.csv"
            )
        else:
            bbox_csv_filepath = root_folder / split / f"{split}-annotations-bbox.csv"
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
        # removing elements with only one target object
        self._remove_single_target_imgs()

        images_with_labels = set(self.box_labels.keys())

        self.images = [
            image_path
            for image_path in all_images
            if image_path.stem in images_with_labels
        ]

        print(f"Loaded dataset from {root_folder}, with {len(self.images)} images.")

        self.label_name_to_class_description = csv_to_dict(
            root_folder / "metadata" / "class-descriptions-boxable.csv",
            discard_header=False,
        )
        self.label_name_to_id = bidict(
            zip(
                self.label_name_to_class_description.keys(),
                range(len(self.label_name_to_class_description.keys())),
            )
        )

        self.transform = transform
        if contextual_distractors:
            self._extract_objects_fn = self._extract_contextual_objects
        else:
            self._extract_objects_fn = self._extract_random_objects

    def __len__(self) -> int:
        return len(self.images)

    @staticmethod
    def unnormalize_coords(coords: List[str], img_shape: Sequence[int]):
        # This method modifies the coordinates list in place
        # coords of format Xmin, Xmax, Ymin, Ymax
        coords = [float(coord) for coord in coords]
        coords[0] *= img_shape[0]
        coords[1] *= img_shape[0]
        coords[2] *= img_shape[1]
        coords[3] *= img_shape[1]
        return coords

    def _remove_single_target_imgs(self):
        keys_to_remove = []
        for key, value in self.box_labels.items():
            if len(value) == 1:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del self.box_labels[key]

    @staticmethod
    def _crop_img(img: torch.Tensor, coords: Sequence[int]):
        top, left = int(coords[3]), int(coords[0])
        height, width = int(coords[3] - coords[2]), int(coords[1] - coords[0])
        return tr_F.crop(img, top=top, left=left, height=height, width=width)

    def _extract_random_objects(self, img, bboxes):
        label_code, *coords = random.choice(bboxes)
        coords = self.unnormalize_coords(coords, img.size)

        img = self._crop_img(img, coords)

        sender_img = self.transform(img, coords)
        receiver_img = self.transform(img, coords)
        label_id = torch.tensor(self.label_name_to_id[label_code])

        return sender_img, label_id, receiver_img, torch.LongTensor([len(bboxes)])

    def _extract_contextual_objects(self, img, bboxes):
        sender_input, receiver_input, label = [], [], []

        for label_code, *coords in bboxes:
            coords = self.unnormalize_coords(coords, img.size)
            img = self._crop_img(img, coords)

            # TODO: for now we only support transforming images after we crop them
            sender_img = self.transform(img, coords)
            receiver_img = self.transform(img, coords)
            label_id = torch.tensor(self.label_name_to_id[label_code])

            sender_input.append(sender_img)
            receiver_input.append(receiver_img)
            label.append(label_id)

        sender_input = torch.stack(sender_input)
        label = torch.stack(label)
        receiver_input = torch.stack(receiver_input)
        num_obj = torch.LongTensor([len(bboxes)])
        return sender_input, label, receiver_input, num_obj

    def __getitem__(self, index: int):
        image_path = self.images[index]
        with open(image_path, "rb") as f:
            image = Image.open(f).convert("RGB")

        labels = self.box_labels[image_path.stem]
        extracted_objects = self._extract_objects_fn(image, labels)

        return extracted_objects


class GaussianBlur:
    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class ImageTransformation:
    def __init__(self, size: int, augmentation: bool = False):
        if augmentation:
            color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
            transformations = [
                transforms.RandomResizedCrop(size=size),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                transforms.RandomHorizontalFlip(),  # with 0.5 probability
                transforms.ToTensor(),
            ]
        else:
            transformations = [
                transforms.Resize(size=(size, size)),
                transforms.ToTensor(),
            ]

        self.transform = transforms.Compose(transformations)

    def __call__(self, img, coords):
        return self.transform(img)
