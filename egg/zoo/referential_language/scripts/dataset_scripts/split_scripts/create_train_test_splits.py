# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# example usage:
#    $ python create_splits.py --dataset_folder="/private/home/rdessi/visual_genome" --train_data=80

import argparse
import json
import random
from pathlib import Path


def load_data(dataset_dir: str, output_folder: str, train_data: int = 80):
    path = Path(dataset_dir)
    path_objects, path_image_data = path / "objects.json", path / "image_data.json"
    output_folder = Path(output_folder)
    train_objs = output_folder / "train_objects.json"
    train_imgs = output_folder / "train_image_data.json"
    val_objs = output_folder / "val_objects.json"
    val_imgs = output_folder / "val_image_data.json"

    with open(path_image_data) as fin:
        img_data = json.load(fin)
    with open(path_objects) as fin:
        obj_data = json.load(fin)

    train_samples = int(len(img_data) * train_data / 100)
    validation_samples = len(img_data) - train_samples
    print(f"Train samples: {train_samples}")
    print(f"Validation samples: {validation_samples}")

    idxs = random.sample(range(len(img_data)), k=len(img_data))

    train_obj_list, train_image_data_list = [], []
    val_obj_list, val_image_data_list = [], []
    for i, idx in enumerate(idxs):
        if i < train_samples:
            train_obj_list.append(obj_data[idx])
            train_image_data_list.append(img_data[idx])
        else:
            val_obj_list.append(obj_data[idx])
            val_image_data_list.append(img_data[idx])

    with open(train_objs, "w") as fout_obj, open(train_imgs, "w") as fout_img:
        json.dump(train_obj_list, fout_obj)
        json.dump(train_image_data_list, fout_img)

    with open(val_objs, "w") as fout_obj, open(val_imgs, "w") as fout_img:
        json.dump(val_obj_list, fout_obj)
        json.dump(val_image_data_list, fout_img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_folder", required=True, help="Path to a folder with VG data"
    )
    parser.add_argument(
        "--output_folder",
        default=None,
        help="Optional, if None, default is dataset_folder",
    )
    parser.add_argument(
        "--train_data",
        type=int,
        default=80,
        help="Proportion of samples used in the train data",
    )
    parser.add_argument("--seed", default=111)
    opts = parser.parse_args()
    random.seed(opts.seed)
    if not opts.output_folder:
        opts.output_folder = opts.dataset_folder

    load_data(
        dataset_dir=opts.dataset_folder,
        output_folder=opts.output_folder,
        train_data=opts.train_data,
    )


if __name__ == "__main__":
    main()
