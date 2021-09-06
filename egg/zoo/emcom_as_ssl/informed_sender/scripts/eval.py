# copyright (c) facebook, inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# NOTE: This script only works with a single gpu

import argparse
import random

import torch
from torchvision import datasets

from egg.zoo.emcom_as_ssl.informed_sender.data import CollaterWithRandomRecv
from egg.zoo.emcom_as_ssl.informed_sender.scripts.utils import (
    add_eval_opts,
    get_game,
    get_params,
)
from egg.zoo.emcom_as_ssl.utils_eval import (
    add_common_cli_args,
    evaluate,
    save_interaction,
    I_TEST_PATH,
    O_TEST_PATH,
)
from egg.zoo.emcom_as_ssl.utils_data import ImageTransformation


class MyRandomSampler(torch.utils.data.sampler.RandomSampler):
    def __init__(self, data_source, game_size):
        self.game_size = game_size
        super(MyRandomSampler, self).__init__(data_source)

    def __iter__(self):
        n = len(self.data_source)
        if self.generator is None:
            generator = torch.Generator()
            generator.manual_seed(
                int(torch.empty((), dtype=torch.int64).random_().item())
            )
        else:
            generator = self.generator
        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from torch.randint(
                    high=n, size=(32,), dtype=torch.int64, generator=generator
                ).tolist()
            yield from torch.randint(
                high=n,
                size=(self.num_samples % 32,),
                dtype=torch.int64,
                generator=generator,
            ).tolist()
        else:
            yield from EvalIterator(n, self.game_size)
            # yield from torch.randperm(n, generator=self.generator).tolist()


class EvalIterator:
    def __init__(self, n, game_size):
        self.n = n
        self._iter = (elem.item() for elem in torch.randperm(n))
        self.game_size = game_size

        self.curr_target_idx = 0
        self.curr_batch = self._generate_batch()

    def __iter__(self):
        return self

    def _generate_batch(self):
        self.curr_target_idx += 1
        curr_batch = [next(self._iter)]
        curr_batch.extend(random.sample(range(self.n), k=self.game_size - 1))
        return (elem for elem in curr_batch)

    def _reset(self):
        self.curr_target_idx = 0
        self.curr_batch = None
        self._iter = (elem.item() for elem in torch.randperm(self.n))

    def __next__(self):
        if self.curr_target_idx >= self.n:
            self._reset()
            raise StopIteration()
        try:
            return next(self.curr_batch)
        except StopIteration:
            self.curr_batch = self._generate_batch()
            return next(self.curr_batch)


def get_dataloader(
    dataset_dir: str,
    image_size: int = 224,
    batch_size: int = 32,
    game_size: int = 2,
    num_workers: int = 6,
    use_augmentations: bool = True,
    seed: int = 111,
):
    transformations = ImageTransformation(image_size, use_augmentations)

    train_dataset = datasets.ImageFolder(dataset_dir, transform=transformations)

    my_sampler = MyRandomSampler(data_source=train_dataset, game_size=game_size)
    collate_fn = CollaterWithRandomRecv(batch_size=batch_size, game_size=game_size)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=my_sampler,
        # shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    return train_loader


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

    if opts.pdb:
        breakpoint()

    print(f"| Loading model from {opts.checkpoint_path} ...")
    game = get_game(opts, opts.checkpoint_path)
    print("| Model loaded.")

    print(f"| Fetching data from {opts.test_set}...")
    if opts.test_set == "o_test":
        dataset_dir = O_TEST_PATH
    elif opts.test_set == "i_test":
        dataset_dir = I_TEST_PATH
    else:
        raise NotImplementedError(f"Cannot recognize {opts.test_set} test_set")

    dataloader = get_dataloader(
        dataset_dir=dataset_dir,
        batch_size=opts.batch_size,
        game_size=opts.game_size,
        use_augmentations=opts.evaluate_with_augmentations,
    )
    print("| Test data fetched.")

    print("| Starting evaluation ...")
    loss, acc, full_interaction = evaluate(game=game, data=dataloader)
    print(f"| Loss: {loss}, accuracy (out of 100): {acc * 100}")

    if opts.dump_interaction_folder:
        save_interaction(
            interaction=full_interaction, log_dir=opts.dump_interaction_folder
        )
        print(f"| Interaction saved at {opts.dump_interaction_folder}")

    print("Finished evaluation.")


if __name__ == "__main__":
    main()
