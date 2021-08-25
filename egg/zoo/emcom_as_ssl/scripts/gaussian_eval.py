# copyright (c) facebook, inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import torch
from torchvision import transforms

from egg.zoo.emcom_as_ssl.scripts.utils import (
    add_common_cli_args,
    evaluate,
    get_game,
    get_params,
)


class GaussianNoiseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        size: int = 49920,
        image_size: int = 224,
        augmentations: bool = False,
    ):
        self.size = size
        self.image_size = image_size
        self.augmentations = augmentations

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        normalize_fn = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        gaussian_sample = torch.randn(3, self.image_size, self.image_size)
        sender_input = normalize_fn(gaussian_sample)

        if self.augmentations:
            gaussian_sample2 = torch.randn_like(gaussian_sample)
            receiver_input = normalize_fn(gaussian_sample2)
        else:
            receiver_input = sender_input.copy()

        return sender_input, torch.zeros(1), receiver_input


def main():
    parser = argparse.ArgumentParser()
    add_common_cli_args(parser)
    cli_args = parser.parse_args()
    opts = get_params(
        shared_vision=cli_args.shared_vision,
        pretrain_vision=cli_args.pretrain_vision,
        vocab_size=cli_args.vocab_size,
    )

    if cli_args.pdb:
        breakpoint()

    print(f"| Loading model from {cli_args.checkpoint_path} ...")
    game = get_game(opts, cli_args.checkpoint_path)
    print("| Model loaded")
    dataset = GaussianNoiseDataset(
        size=49920, augmentations=cli_args.evaluate_with_augmentations
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    print("| Starting evaluation ...")
    loss, accuracy, _ = evaluate(game=game, data=dataloader)
    print(f"| Loss: {loss}, accuracy (out of 100)")


if __name__ == "__main__":
    main()
