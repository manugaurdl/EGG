# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from pathlib import Path

import torch
import torchvision

import egg.core as core
from egg.zoo.emcom_as_ssl.data import get_dataloader
from egg.zoo.emcom_as_ssl.games import build_game
from egg.zoo.emcom_as_ssl.utils import add_weight_decay, get_common_opts


def main(params):
    opts = get_common_opts(params=params)
    print(f"{opts}\n")

    if not opts.distributed_context.is_distributed and opts.pdb:
        breakpoint()

    train_loader = get_dataloader(
        dataset_dir=opts.dataset_dir,
        dataset_name=opts.dataset_name,
        image_size=opts.image_size,
        batch_size=opts.batch_size,
        num_workers=opts.num_workers,
        is_distributed=opts.distributed_context.is_distributed,
        seed=opts.random_seed,
        use_augmentations=opts.use_augmentations,
        return_original_image=opts.return_original_image,
    )

    game = build_game(opts)

    model_parameters = add_weight_decay(game, opts.weight_decay, skip_name="bn")

    optimizer = torch.optim.SGD(
        model_parameters,
        lr=opts.lr,
        momentum=0.9,
    )
    optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=opts.n_epochs
    )

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        optimizer_scheduler=optimizer_scheduler,
        train_data=train_loader,
        callbacks=[core.ConsoleLogger(as_json=True, print_train_loss=True)],
    )

    data_args = {
        "image_size": opts.image_size,
        "batch_size": opts.batch_size,
        "dataset_name": "imagenet",
        "num_workers": opts.num_workers,
        "use_augmentations": False,
        "is_distributed": opts.distributed_context.is_distributed,
        "seed": opts.random_seed,
    }
    i_test_loader = get_dataloader(
        dataset_dir="/datasets01/imagenet_full_size/061417/val", **data_args
    )

    _, i_test_interaction = trainer.eval(i_test_loader)

    max_items = 5000

    tmp_input = [
        torchvision.transforms.functional.resize(x, [128, 128])
        for x in i_test_interaction.sender_input[:max_items]
    ]

    inp = torch.stack(tmp_input, dim=0)

    i_test_interaction.sender_input = inp
    i_test_interaction.message = i_test_interaction.message[:max_items]

    dump = dict((k, v.mean().item()) for k, v in i_test_interaction.aux.items())
    dump.update(dict(mode="VALIDATION_I_TEST"))
    print(json.dumps(dump), flush=True)

    if opts.checkpoint_dir:
        output_path = Path(opts.checkpoint_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        torch.save(i_test_interaction, output_path / "i_test_interaction")

    print("| FINISHED JOB")


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    import sys

    main(sys.argv[1:])
