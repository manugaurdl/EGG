# copyright (c) facebook, inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import torch
from torchvision import transforms

from egg.zoo.emcom_as_ssl.informed_sender.scripts.utils import (
    add_eval_opts,
    get_game,
    get_params,
)
from egg.zoo.emcom_as_ssl.utils_eval import (
    add_common_cli_args,
    evaluate,
)


def collate_with_random_recv_input(batch):
    sender_input, receiver_input, class_labels = [], [], []
    for elem in batch:
        sender_input.append(elem[0])
        receiver_input.append(elem[2])
        class_labels.append(torch.LongTensor([elem[1]]))

    bsz = len(batch)
    sender_input = torch.stack(sender_input)
    receiver_input = torch.stack(receiver_input)
    class_labels = torch.stack(class_labels).view(2, 1, bsz // 2, -1)

    random_order = torch.stack([torch.randperm(bsz // 2) for _ in range(2)])
    target_position = torch.argmin(random_order, dim=1)

    return (
        sender_input,
        class_labels,
        receiver_input,
        {"target_position": target_position, "random_order": random_order},
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
            receiver_input = sender_input.detach()

        return sender_input, 1, receiver_input


def main():
    parser = add_common_cli_args()
    add_eval_opts(parser)
    cli_args = parser.parse_args()
    args = get_params(
        shared_vision=cli_args.shared_vision,
        pretrain_vision=cli_args.pretrain_vision,
        vocab_size=cli_args.vocab_size,
        batch_size=cli_args.batch_size,
        game_size=cli_args.game_size,
    )

    cli_args, args = vars(cli_args), vars(args)
    opts = argparse.Namespace(**{**cli_args, **args})

    if cli_args.pdb:
        breakpoint()

    print(f"| Loading model from {cli_args.checkpoint_path} ...")
    game = get_game(opts, cli_args.checkpoint_path)
    print("| Model loaded")
    data_size = 49920 * cli_args.repeat

    dataset = GaussianNoiseDataset(
        size=data_size, augmentations=cli_args.evaluate_with_augmentations
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cli_args.batch_size,
        num_workers=6,
        collate_fn=collate_with_random_recv_input,
        pin_memory=True,
        drop_last=True,
    )

    print("| Starting evaluation ...")
    loss, accuracy, _ = evaluate(game=game, data=dataloader)
    print(f"| Loss: {loss}, {round(accuracy * 100, 2)} accuracy (out of 100)")


if __name__ == "__main__":
    main()
