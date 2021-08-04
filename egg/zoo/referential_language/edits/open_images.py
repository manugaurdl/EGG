# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import pathlib

import cv2


class OpenImagesDataset:
    def __init__(
        self,
        root,
        dataset_type="train",
    ):
        self.root = pathlib.Path(root)
        self.dataset_type = dataset_type.lower()

        self.data, self.class_names, self.class_dict = self._read_data()
        self.min_image_num = -1
        self.ids = [info["image_id"] for info in self.data]

        self.class_stat = None

    def __getitem__(self, index):
        _ = f"{self.root}/{self.dataset_type}-annotations-bbox.csv"
        image_info = self.data[index]
        image = self._read_image(image_info["image_id"])
        # duplicate boxes to prevent corruption of dataset
        boxes = copy.copy(image_info["boxes"])
        boxes[:, 0] *= image.shape[1]
        boxes[:, 1] *= image.shape[0]
        boxes[:, 2] *= image.shape[1]
        boxes[:, 3] *= image.shape[0]
        # duplicate labels to prevent corruption of dataset
        labels = copy.copy(image_info["labels"])

        # return image_info["image_id"], image, boxes, labels
        return image, boxes, labels

    def get_image(self, index):
        image_info = self.data[index]
        image_id = image_info["image_id"]
        image_file = self.root / self.dataset_type / f"{image_id}.jpg"
        image = cv2.imread(str(image_file))
        if image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image


def main():
    breakpoint()
    o = OpenImagesDataset(
        root="/datasets01/open_images/030119/validation", dataset_type="validation"
    )
    breakpoint()
    return o


if __name__ == "__main__":
    main()
