# NOTE: This script only works with a single gpu

import argparse
from typing import Any, List

import torch
from torchvision import datasets

from egg.zoo.emcom_as_ssl.new_dataloaders import ImageTransformation
from egg.zoo.emcom_as_ssl.scripts.utils import (
    add_common_cli_args,
    evaluate,
    get_game,
    get_params,
    I_TEST_PATH,
    O_TEST_PATH,
    save_interaction,
)


class Collater:
    def __init__(
        self,
        batch_size=128,
        distractors=1,
        transformations=None,
    ):
        self.batch_size = batch_size
        assert (
            self.batch_size % 2 == 0
        ), f"batch_size my be a multiple of 2, found {batch_size} instead"
        self.distractors = distractors
        assert self.distractors == 1, "currently only one distractors is supported"
        self.transformations = transformations

    def __call__(self, batch: List[Any]):
        sender_input, receiver_input, all_class_labels = [], [], []
        for elem in batch:
            sender_input.append(elem[0][0])
            receiver_input.append(elem[0][1])
            all_class_labels.append(torch.LongTensor([elem[1]]))

        sender_input = sender_input.repeat(128, 1, 1, 1).view(128, 128, 1, 1, 1)
        receiver_input = receiver_input.repeat(128, 1, 1, 1).view(128, 128, 1, 1, 1)

        recv_input_order = torch.stack(
            [torch.randperm(self.distractors + 1) for _ in range(self.batch_size)]
        )

        for idx in range(self.batch_size):
            receiver_input[idx, torch.arange(self.distractors + 1)] = receiver_input[
                idx, recv_input_order[idx]
            ]
        targets_position = torch.stack(
            [
                torch.argmax((recv_input_order[idx] == idx).int())
                for idx in range(self.batch_size)
            ]
        )

        class_labels = torch.cat(
            [all_class_labels[targets_position[idx]] for idx in range(self.batch_size)]
        )

        return sender_input, (class_labels, targets_position), receiver_input


def get_dataloader(
    dataset_dir: str,
    image_size: int = 224,
    batch_size: int = 128,
    n_distractors: int = 1,
    num_workers: int = 4,
    use_augmentations: bool = True,
    seed: int = 111,
):
    print(f"using {n_distractors} distractors")
    transformations = ImageTransformation(image_size, use_augmentations)

    train_dataset = datasets.ImageFolder(dataset_dir, transform=transformations)

    collater = Collater(
        batch_size=batch_size,
        distractors=n_distractors,
        transformations=transformations,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collater,
        pin_memory=True,
        drop_last=True,
    )

    return train_loader


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
    print("| Model loaded.")

    if cli_args.test_set == "o_test":
        dataset_dir = O_TEST_PATH
    elif cli_args.test_set == "i_test":
        dataset_dir = I_TEST_PATH
    else:
        raise NotImplementedError(f"Cannot recognize {cli_args.test_set} test_set")

    print(f"| Fetching data for {cli_args.test_set} test set from {dataset_dir}...")
    dataloader = get_dataloader(
        n_distractors=127,
        dataset_dir=dataset_dir,
        use_augmentations=cli_args.evaluate_with_augmentations,
    )
    print("| Test data fetched.")

    print("| Starting evaluation ...")
    loss, soft_acc, game_acc, full_interaction = evaluate(game=game, data=dataloader)
    print(
        f"| Loss: {loss}, soft_accuracy (out of 100): {soft_acc * 100}, game_accuracy (out of 100): {game_acc * 100}"
    )

    if cli_args.dump_interaction_folder:
        save_interaction(
            interaction=full_interaction,
            log_dir=cli_args.dump_interaction_folder,
            test_set=cli_args.test_set,
        )
        print(f"| Interaction saved at {cli_args.dump_interaction_folder}")

    print("Finished evaluation.")


if __name__ == "__main__":
    main()
