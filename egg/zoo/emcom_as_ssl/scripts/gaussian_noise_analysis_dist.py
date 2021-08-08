# copyright (c) facebook, inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# python -m egg.zoo.emcom_as_ssl.scripts.gaussian_noise_analysis \
#    --loss_type="xent" \
#    --checkpoint_path="<path_to_checkpoint_folder>/final.tar" \


import argparse

import torch

from egg.zoo.emcom_as_ssl.scripts.utils import (
    add_common_cli_args,
    evaluate,
    get_game,
    get_params,
)


def get_random_noise_dataloader(
    dataset_size: int = 49152,
    batch_size: int = 128,
    image_size: int = 224,
    num_workers: int = 4,
    use_augmentations: bool = False,
):

    dataset = GaussianNoiseDataset(size=dataset_size, image_size=image_size)

    # TODO
    collater = Collater()

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collater,
        pin_memory=True,
        drop_last=True,
    )
    return loader


class Collater:
    def __init__(self):
        self.distractors = 1

    def __call__(self, batch):
        assert (
            len(batch) % 2 == 0
        ), f"batch_size must be a multiple of 2, found {len(batch)} instead"
        batch_size = len(batch)

        # this piece of code does not work with self.distractors > 1
        targets_position = torch.randint(self.distractors + 1, size=(batch_size // 2,))

        sender_input, receiver_input, class_labels = [], [], []
        for elem in batch:
            sender_input.append(elem[0][0])
            receiver_input.append(elem[0][1])
            class_labels.append(torch.LongTensor([elem[1]]))

        img_size = sender_input[0].shape

        sender_input = torch.stack(sender_input).view(
            batch_size // 2, self.distractors + 1, *img_size
        )
        sender_input = sender_input[torch.arange(batch_size // 2), targets_position]

        receiver_input = torch.stack(receiver_input).view(
            batch_size // 2, self.distractors + 1, *img_size
        )

        class_labels = torch.stack(class_labels).view(-1, self.distractors + 1)
        class_labels = class_labels[torch.arange(batch_size // 2), targets_position]

        return sender_input, (class_labels, targets_position), receiver_input


class GaussianNoiseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        size: int = 3072,
        image_size: int = 224,
    ):
        self.image_size = image_size
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        sender_input = torch.randn(3, self.image_size, self.image_size)
        receiver_input = torch.randn(3, self.image_size, self.image_size)
        return (sender_input, receiver_input), torch.zeros(1)


def main():
    parser = argparse.ArgumentParser()
    add_common_cli_args(parser)
    cli_args = parser.parse_args()
    opts = get_params(
        simclr_sender=cli_args.simclr_sender,
        shared_vision=cli_args.shared_vision,
        loss_type=cli_args.loss_type,
        discrete_evaluation_simclr=cli_args.discrete_evaluation_simclr,
        vocab_size=cli_args.vocab_size,
    )

    if cli_args.pdb:
        breakpoint()

    print(f"| Loading model from {cli_args.checkpoint_path} ...")
    game = get_game(opts, cli_args.checkpoint_path)
    print("| Model loaded")
    dataloader = get_random_noise_dataloader()

    print("| Starting evaluation ...")
    loss, soft_acc, game_acc, _ = evaluate(game=game, data=dataloader)
    print(
        f"| Loss: {loss}, soft_accuracy (out of 100): {soft_acc * 100}, game_accuracy (out of 100): {game_acc * 100}"
    )


if __name__ == "__main__":
    main()
