# copyright (c) facebook, inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# NOTE: This script only works with a single gpu

import argparse

from egg.zoo.emcom_as_ssl.linear_sender.data import get_dataloader
from egg.zoo.emcom_as_ssl.linear_sender.scripts.utils import get_game, get_params
from egg.zoo.emcom_as_ssl.utils_eval import (
    add_common_cli_args,
    evaluate,
    save_interaction,
    I_TEST_PATH,
    O_TEST_PATH,
)


def main():
    parser = add_common_cli_args()
    cli_args, _ = parser.parse_known_args()
    args = get_params(
        shared_vision=cli_args.shared_vision,
        pretrain_vision=cli_args.pretrain_vision,
        vocab_size=cli_args.vocab_size,
        batch_size=cli_args.batch_size,
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
        use_augmentations=opts.evaluate_with_augmentations,
        is_distributed=opts.distributed_context.is_distributed,
    )
    print("| Test data fetched.")

    print("| Starting evaluation ...")
    loss, acc, full_interaction = evaluate(game=game, data=dataloader)
    print(f"| Loss: {loss}, accuracy (out of 100): {round(acc * 100, 2)}")

    if opts.dump_interaction_folder:
        save_interaction(
            interaction=full_interaction, log_dir=opts.dump_interaction_folder
        )
        print(f"| Interaction saved at {opts.dump_interaction_folder}")

    print("Finished evaluation.")


if __name__ == "__main__":
    main()
