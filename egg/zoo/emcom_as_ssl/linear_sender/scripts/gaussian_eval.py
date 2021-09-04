# copyright (c) facebook, inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import torch
from torchvision import transforms

from egg.zoo.emcom_as_ssl.linear_sender.scripts.utils import get_game, get_params
from egg.zoo.emcom_as_ssl.utils_eval import add_common_cli_args, evaluate


class GaussianNoiseDataset(torch.utils.data.Dataset):
    def __init__(self, size: int = 49920, image_size: int = 224):
        self.size = size
        self.image_size = image_size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        normalize_fn = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        gaussian_sample = torch.randn(3, self.image_size, self.image_size)
        sender_input = normalize_fn(gaussian_sample)
        receiver_input = sender_input.detach()

        return sender_input, torch.zeros(1), receiver_input


def main():
    parser = add_common_cli_args()
    cli_args, _ = parser.parse_known_args()
    args = get_params(
        shared_vision=cli_args.shared_vision,
        pretrain_vision=cli_args.pretrain_vision,
        vocab_size=cli_args.vocab_size,
    )
    cli_args, args = vars(cli_args), vars(args)
    opts = argparse.Namespace(**{**cli_args, **args})

    if opts.pdb:
        breakpoint()

    print(f"| Loading model from {opts.checkpoint_path} ...")
    game = get_game(opts, opts.checkpoint_path)
    print("| Model loaded")
    dataset = GaussianNoiseDataset()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        num_workers=6,
        pin_memory=True,
        drop_last=True,
    )

    print("| Starting evaluation ...")
    loss, accuracy, _ = evaluate(game=game, data=dataloader)
    print(f"| Loss: {loss}, {round(accuracy * 100, 2)} accuracy (out of 100)")


if __name__ == "__main__":
    main()
