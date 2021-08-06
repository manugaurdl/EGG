# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import pathlib
from PIL import Image, ImageFilter
import random

import numpy as np
import torch
import torchvision.transforms as transforms


class _BatchIterator:
    def __init__(
        self,
        dataset,
        batch_size=128,
        distractors=1,
        max_targets_seen=-1,
        transformations=None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.distractors = distractors
        self.max_targets_seen = (
            max_targets_seen if max_targets_seen > 0 else len(dataset)
        )

        self.targets_seen = 0
        self.transformations = transformations

    def __iter__(self):
        return self

    def __next__(self):
        if self.targets_seen + self.batch_size > self.max_targets_seen:
            self.targets_seen = 0  # implement drop_last
            raise StopIteration()
        lineup_size = self.batch_size * (self.distractors + 1)
        sampled_idxs = np.random.choice(range(len(self.dataset)), size=lineup_size)
        targets_position = torch.from_numpy(
            np.random.randint(self.distractors + 1, size=self.batch_size)
        )
        # dataset returns a tuple of two Tensors, one with the image and one with class_id
        receiver_input = torch.stack(
            [self.transformations(self.dataset[idx][0]) for idx in sampled_idxs]
        )

        img_size = receiver_input[0].shape[-1]
        receiver_input = receiver_input.view(
            self.batch_size, self.distractors + 1, 3, img_size, img_size
        )

        # start of very dirty hack
        receiver_input_for_sender = torch.stack(
            [self.transformations(self.dataset[idx][0]) for idx in sampled_idxs]
        )
        receiver_input_for_sender = receiver_input_for_sender.view(
            self.batch_size, self.distractors + 1, 3, img_size, img_size
        )
        # end of very dirty hack

        class_labels = torch.cat([self.dataset[idx][1] for idx in sampled_idxs])
        class_labels = class_labels.view(self.batch_size, self.distractors + 1)

        targets = []
        for idx in range(self.batch_size):
            targets.append(
                receiver_input_for_sender[idx, targets_position[idx], :, :, :]
            )

        # receiver_input = receiver_input.view(
        #    self.batch_size * (self.distractors + 1), 3, 224, 224
        # )

        targets = torch.stack(targets)
        self.targets_seen += self.batch_size

        return (targets, receiver_input), class_labels, targets_position


class ImagenetDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        super(ImagenetDataset, self).__init__()

        with open(data_path) as fd:
            lines = fd.readlines()
            if lines[-1] == "":
                lines = lines[:-1]  # removing last newline from file
        self.paths = lines

        # classes = sorted(elem.split("/")[-2] for elem in lines)
        classes = sorted(list(set(elem.split("/")[-2] for elem in lines)))

        # checks
        original_class = torch.load(
            "/private/home/rdessi/scripts/class_to_idx_original.pt"
        )
        assert len(classes) == len(original_class)
        for cls_name in classes:
            assert cls_name in original_class

        self.class_to_idx = {k: i for i, k in enumerate(classes)}

        for cls_name, idx in self.class_to_idx.items():
            assert idx == original_class[cls_name]

        """
        self.idx_to_class = {i: k for i, k in enumerate(classes)}
        torch.save(
            self.idx_to_class,
            "/private/home/rdessi/scripts/idx_to_class_dataset_with_distractor.pt",
        )
        """

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx].strip()
        img = Image.open(img_path).convert("RGB")
        img.load()

        label = self.class_to_idx[img_path.split("/")[-2]]
        return img, torch.Tensor([label])


class ImagenetDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        self.distractors = kwargs.pop("distractors")
        self.max_targets_seen = kwargs.pop("max_targets_seen")
        self.transformations = kwargs.pop("transformations")
        super(ImagenetDataLoader, self).__init__(*args, **kwargs)

    def __iter__(self):
        return _BatchIterator(
            self.dataset,
            self.batch_size,
            self.distractors,
            self.max_targets_seen,
            transformations=self.transformations,
        )


def get_dataloader(
    dataset_dir: str,
    image_size: int = 224,
    batch_size: int = 128,
    n_distractors: int = 1,
    max_targets_seen: int = -1,
    num_workers: int = 4,
    use_augmentations: bool = True,
    is_distributed: bool = False,
    return_original_image: bool = False,
    seed: int = 111,
):

    print(f"using {n_distractors} distractors with {max_targets_seen} targets_seen")

    train_path = pathlib.Path(dataset_dir)

    transformations = ImageTransformation(
        image_size, use_augmentations, return_original_image
    )

    train_dataset = ImagenetDataset(data_path=train_path)

    train_sampler = None
    if is_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=True, drop_last=True, seed=seed
        )

    train_loader = ImagenetDataLoader(
        dataset=train_dataset,
        distractors=n_distractors,
        max_targets_seen=max_targets_seen,
        transformations=transformations,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return train_loader


class GaussianBlur:
    """Gaussian blur augmentation as in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class ImageTransformation:
    """
    A stochastic data augmentation module that transforms any given data example
    randomly resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(
        self, size: int, augmentation: bool = False, return_original_image: bool = False
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

        transformations.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.transform = transforms.Compose(transformations)

        self.return_original_image = return_original_image

    def __call__(self, x):
        return self.transform(x)
