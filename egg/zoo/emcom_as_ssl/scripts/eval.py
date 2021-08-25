# copyright (c) facebook, inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# NOTE: This script only works with a single gpu

import argparse

from egg.zoo.emcom_as_ssl.scripts.utils import (
    add_common_cli_args,
    evaluate,
    get_dataloader,
    get_game,
    get_params,
    save_interaction,
)

O_TEST_PATH = (
    "/private/home/mbaroni/agentini/representation_learning/"
    "generalizaton_set_construction/80_generalization_data_set/"
)
I_TEST_PATH = "/datasets01/imagenet_full_size/061417/val"


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
        image_size=opts.image_size,
        batch_size=opts.batch_size,
        num_workers=opts.num_workers,
        use_augmentations=cli_args.evaluate_with_augmentations,
        is_distributed=False,
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
