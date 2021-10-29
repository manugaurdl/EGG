# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random

from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence

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

        sender_input = torch.cat(sender_input)
        labels = torch.cat(labels)
        receiver_input = torch.cat(receiver_input)
        aux_input = {"idx": torch.cat(elem_idx_in_batch)}

        assert max_batch_size == sender_input.shape[0]
        assert max_batch_size == labels.shape[0]
        assert max_batch_size == receiver_input.shape[0]

        return sender_input, labels, receiver_input, aux_input

    def __call__(self, batch: List[Any]) -> List[torch.Tensor]:
        return self.collate_fn(batch)


def get_dataloader(
    dataset_dir: str,
    split: str,
    batch_size: int = 128,
    num_workers: int = 4,
    contextual_distractors: bool = False,
    image_size: int = 64,
    shuffle: bool = True,
    use_augmentations: bool = True,
    is_distributed: bool = False,
    seed: int = 111,
):

    transform = ImageTransformation(size=image_size, augmentation=use_augmentations)

    dataset = OpenImages(
        split=split,
        transform=transform,
        contextual_distractors=contextual_distractors,
    )

    sampler = None
    if is_distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=shuffle, drop_last=True, seed=seed
        )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=Collater(contextual_distractors=contextual_distractors),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader


class ItOpenImagesDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        split: str,
        world_size: int,
        rank: int,
        transform: Optional[Callable] = None,
    ):
        root_folder = Path("/datasets01/open_images/030119")

        all_images_path = (
            f"/private/home/rdessi/contextual_emcomm/{split}_path_files.txt"
        )

        bbox_csv_filepath = (
            f"/private/home/rdessi/contextual_emcomm/{split}-annotations-bbox.csv"
        )
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

        with open(all_images_path) as fin:
            self.images = [
                Path(image_path.strip())
                for image_path in fin
                if Path(image_path.strip()).stem in images_with_labels
            ]

        print(f"Loaded dataset from {root_folder}, with {len(self.images)} images.")

        self.label_name_to_id = bidict(
            zip(
                self.label_name_to_class_description.keys(),
                range(len(self.label_name_to_class_description.keys())),
            )
        )

        self.transform = transform

        n_samples_per_gpu = len(self.images) // world_size
        self.start = rank * n_samples_per_gpu
        self.end = self.start + n_samples_per_gpu

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            for i in range(self.start, self.end):
                yield (i)

    def __next__(self):
        pass


class OpenImages(VisionDataset):
    def __init__(
        self,
        split: str = "train",
        transform: Optional[Callable] = None,
        contextual_distractors: bool = False,
    ):
        root_folder = Path("/datasets01/open_images/030119")
        super(OpenImages, self).__init__(root=root_folder, transform=transform)

        all_images_path = (
            f"/private/home/rdessi/contextual_emcomm/{split}_path_files.txt"
        )

        bbox_csv_filepath = (
            f"/private/home/rdessi/contextual_emcomm/{split}-annotations-bbox.csv"
        )
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

        with open(all_images_path) as fin:
            self.images = [
                Path(image_path.strip())
                for image_path in fin
                if Path(image_path.strip()).stem in images_with_labels
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
        top, left = int(coords[2]), int(coords[0])
        height = max(1, int(coords[3] - coords[2]))
        width = max(1, int(coords[1] - coords[0]))
        return tr_F.crop(img, top=top, left=left, height=height, width=width)

    def _extract_random_objects(self, img, bboxes):
        label_code, *coords = random.choice(bboxes)
        coords = self.unnormalize_coords(coords, img.size)

        img = self._crop_img(img, coords)

        sender_img, receiver_img = self.transform(img), self.transform(img)
        label_id = torch.tensor(self.label_name_to_id[label_code])

        return sender_img, label_id, receiver_img, torch.LongTensor([len(bboxes)])

    def _extract_contextual_objects(self, img, bboxes):
        sender_input, receiver_input, label = [], [], []

        for label_code, *coords in bboxes:
            coords = self.unnormalize_coords(coords, img.size)
            if len(bboxes) > 2:
                width = int(coords[1] - coords[0])
                height = int(coords[3] - coords[2])
                if width * height < 20:
                    continue
            img = self._crop_img(img, coords)

            sender_img, receiver_img = self.transform(img), self.transform(img)
            label_id = torch.tensor(self.label_name_to_id[label_code])

            sender_input.append(sender_img)
            receiver_input.append(receiver_img)
            label.append(label_id)

        sender_input = torch.stack(sender_input)
        label = torch.stack(label)
        receiver_input = torch.stack(receiver_input)
        num_obj = torch.LongTensor([len(sender_input)])
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
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                transforms.RandomHorizontalFlip(),  # with 0.5 probability
                transforms.Resize(size=(size, size)),
                transforms.ToTensor(),
            ]
        else:
            transformations = [
                transforms.Resize(size=(size, size)),
                transforms.ToTensor(),
            ]
        self.transform = transforms.Compose(transformations)

    def __call__(self, img):
        return self.transform(img)
