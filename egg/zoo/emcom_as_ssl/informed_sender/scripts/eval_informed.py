# copyright (c) facebook, inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# NOTE: This script only works with a single gpu

import argparse

import torch
from torchvision import datasets

from egg.zoo.emcom_as_ssl.data import (
    ImageTransformation,
    collate_with_random_recv_input,
)
from egg.zoo.emcom_as_ssl.scripts.utils import (
    add_common_cli_args,
    evaluate,
    get_game,
    get_params,
    save_interaction,
)

O_TEST_PATH = (
    "/private/home/mbaroni/agentini/representation_learning/"
    "generalizaton_set_construction/80_generalization_data_set/"
)
I_TEST_PATH = "/datasets01/imagenet_full_size/061417/val"


class MyImageFolder(datasets.ImageFolder):
    def update_images(self, repeat):
        if repeat > 1:
            self.imgs *= repeat


def get_dataloader(
    dataset_dir: str,
    repeat: int = 2,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 0,
    use_augmentations: bool = True,
    seed: int = 111,
):
    transformations = ImageTransformation(image_size, use_augmentations)

    train_dataset = MyImageFolder(dataset_dir, transform=transformations)
    train_dataset.update_images(repeat)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_with_random_recv_input,
        pin_memory=True,
        drop_last=True,
    )

    return train_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repeat",
        type=int,
        default=2,
        help="How many times the dataset will be repeated. Useful for balancing the batching+distractors",
    )
    parser.add_argument("--force_compare_two", default=False, action="store_true")
    add_common_cli_args(parser)
    cli_args = parser.parse_args()
    cli_args.informed_sender = True

    opts = get_params(**vars(cli_args))

    if cli_args.pdb:
        breakpoint()

    print(f"| Loading model from {cli_args.checkpoint_path} ...")
    game = get_game(opts, cli_args.checkpoint_path)
    print("| Model loaded.")

    print(f"| Fetching data from {cli_args.test_set}...")
    if cli_args.test_set == "o_test":
        dataset_dir = O_TEST_PATH
    elif cli_args.test_set == "i_test":
        dataset_dir = I_TEST_PATH
    else:
        raise NotImplementedError(f"Cannot recognize {cli_args.test_set} test_set")

    dataloader = get_dataloader(
        dataset_dir=dataset_dir,
        batch_size=cli_args.batch_size,
        repeat=cli_args.repeat,
        use_augmentations=cli_args.evaluate_with_augmentations,
    )
    print("| Test data fetched.")

    print("| Starting evaluation ...")
    loss, acc, full_interaction = evaluate(game=game, data=dataloader)
    print(f"| Loss: {loss}, accuracy (out of 100): {acc * 100}")

    if cli_args.dump_interaction_folder:
        save_interaction(
            interaction=full_interaction, log_dir=cli_args.dump_interaction_folder
        )
        print(f"| Interaction saved at {cli_args.dump_interaction_folder}")

    print("Finished evaluation.")


if __name__ == "__main__":
    main()
