# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AdamW, get_linear_schedule_with_warmup

import egg.core as core
from egg.core import Callback, ConsoleLogger, Interaction
from egg.zoo.contextual_game.game import (
    ClipClapSenderTrain,
    Game,
    convert_models_to_fp32,
)
from egg.zoo.contextual_game.data import get_dataloader
from egg.zoo.contextual_game.opts import get_common_opts
from egg.zoo.contextual_game.utils import (
    get_sha,
    store_job_and_task_id,
)


def print_grad_info(model):
    grad, no_grad = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            no_grad.append(name)
            continue
        grad.append(name)
    print(f"GRAD {grad}")
    # print(f"NO GRAD {no_grad}")


class ModelSaver(Callback):
    def save_clipclap_model(self, epoch=""):
        if hasattr(self.trainer, "checkpoint_path"):
            if (
                self.trainer.checkpoint_path
                and self.trainer.distributed_context.is_leader
            ):
                self.trainer.checkpoint_path.mkdir(exist_ok=True, parents=True)
                model_name = f"clip_clap_model_{epoch if epoch else 'final'}.pt"
                torch.save(
                    self.trainer.game.sender.clipclap_model.state_dict(),
                    self.trainer.checkpoint_path / model_name,
                )

    def on_epoch_end(self, loss: float, _logs: Interaction, epoch: int):
        self.save_clipclap_model(epoch=epoch)

    def on_train_end(self):
        self.save_clipclap_model()


class Loss:
    def __init__(self, prefix_len):
        self.prefix_len = prefix_len

    def __call__(
        self,
        _sender_input,
        message,
        _receiver_input,
        _receiver_output,
        _labels,
        aux_input,
    ):
        caption = aux_input["captions"].tolist()[0]
        gt = torch.Tensor([token for token in caption if token >= 0])
        gt = gt.long().to(message.device)

        logits = message[:, self.prefix_len - 1 : -1]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), gt.flatten())
        return loss, {}


def build_game(opts):
    clip_model = clip.load(opts.clip_model)[0]
    convert_models_to_fp32(clip_model)

    sender = ClipClapSenderTrain(
        model_path=opts.clipclap_model_path,
        mapping_type=opts.mapping_type,
        constant_prefix_tokens=opts.constant_prefix_tokens,
        clip_prefix_tokens=opts.clip_prefix_tokens,
        clip_prefix_size=clip_model.visual.output_dim,
        num_layers=opts.num_transformer_layers,
        clip_model=opts.clip_model,
        use_beam_search=opts.use_beam_search,
        num_beams=opts.num_beams,
        prefix_only=opts.prefix_only,
    )

    class DummyRecv(nn.Module):
        def forward(self, message, receiver_input, aux_input=None):
            return torch.zeros(1)

    game = Game(sender, DummyRecv(), Loss(opts.constant_prefix_tokens))
    game.train()
    return game


def main(params):
    start = time.time()
    opts = get_common_opts(params=params)

    store_job_and_task_id(opts)
    print(opts)
    print(get_sha())

    if not opts.distributed_context.is_distributed and opts.debug:
        breakpoint()

    train_loader = get_dataloader(
        image_dir=opts.image_dir,
        metadata_dir=opts.metadata_dir,
        batch_size=opts.batch_size,
        image_size=opts.image_size,
        split="train",
        num_workers=opts.num_workers,
        is_distributed=opts.distributed_context.is_distributed,
        seed=opts.random_seed,
    )

    game = build_game(opts)
    print_grad_info(game)

    optimizer = AdamW(game.parameters(), lr=opts.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=opts.warmup_steps,
        num_training_steps=opts.n_epochs * len(train_loader),
    )

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        optimizer_scheduler=scheduler,
        train_data=train_loader,
        callbacks=[ConsoleLogger(as_json=True, print_train_loss=True), ModelSaver()],
        debug=opts.debug,
    )

    trainer.train(opts.n_epochs)

    end = time.time()
    print(f"| Run took {end - start:.2f} seconds")
    print("| FINISHED JOB")


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    torch.set_deterministic(True)
    import sys

    main(sys.argv[1:])
