# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
from typing import List

import torch

import wandb
from egg.core import Callback, ConsoleLogger, Interaction


class BestStatsTracker(Callback):
    def __init__(self):
        self.best = {"acc": -float("inf"), "loss": float("inf"), "epoch": -1}

    def on_epoch_end(self, loss, logs: Interaction, epoch: int):
        if logs.aux["acc"].mean().item() > self.best["acc"]:
            self.best["acc"] = logs.aux["acc"].mean().item()
            self.best["loss"] = loss
            self.best["epoch"] = epoch

    def on_train_end(self):
        best_stats = dict(mode="best_stats", **self.best)
        print(json.dumps(best_stats), flush=True)


class VisionModelSaver(Callback):
    """A callback that stores vision module(s) in trainer's checkpoint_dir, if any."""

    def __init__(self, checkpoint_freq: int = 1):
        self.checkpoint_freq = checkpoint_freq

    def on_train_begin(self, trainer_instance: "Trainer"):  # noqa: F821
        self.trainer = trainer_instance

        if self.trainer.distributed_context.is_distributed:
            # if distributed training the model is an instance of
            # DistributedDataParallel and we need to unpack it from it.
            self.vision_module = self.trainer.game.module.vision_module
        else:
            self.vision_module = self.trainer.game.vision_module
        self.shared = self.vision_module.shared

    def write_model(self, agent_cnn: torch.nn.Module, model_name: str):
        torch.save(
            agent_cnn.state_dict(),
            self.trainer.checkpoint_path / model_name,
        )

    def save_vision_model(self, epoch=""):
        if hasattr(self.trainer, "checkpoint_path") and self.trainer.checkpoint_path:
            self.trainer.checkpoint_path.mkdir(exist_ok=True, parents=True)

            model_name = f"vision_module_{'shared' if self.shared else 'sender'}_{epoch if epoch else 'final'}.pt"
            self.write_model(self.vision_module.encoder, model_name)

            if not self.shared:
                model_name = f"vision_module_recv_{epoch if epoch else 'final'}.pt"
                self.write_model(self.vision_module.encoder_recv, model_name)

    def on_train_end(self):
        if self.trainer.distributed_context.is_leader:
            self.save_vision_model()

    def on_epoch_end(self, loss: float, _logs: Interaction, epoch: int):
        if self.checkpoint_freq > 0 and (epoch % self.checkpoint_freq == 0):
            if self.trainer.distributed_context.is_leader:
                self.save_vision_model(epoch=epoch)


class DistributedSamplerEpochSetter(Callback):
    """A callback that sets the right epoch of a DistributedSampler instance."""

    def on_epoch_begin(self, epoch: int):
        if self.trainer.distributed_context.is_distributed:
            self.trainer.train_data.sampler.set_epoch(epoch)

    def on_test_begin(self, epoch: int):
        if self.trainer.distributed_context.is_distributed:
            self.trainer.validation_data.sampler.set_epoch(epoch)


class WandbLogger(Callback):
    def on_batch_end(
        self, logs: Interaction, loss: float, batch_id: int, is_training: bool = True
    ):
        if self.trainer.distributed_context.is_leader:
            wandb.log(
                {
                    "batch_loss": loss,
                    "batch_accuracy": logs.aux["acc"].mean().item(),
                }
            )

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        if self.trainer.distributed_context.is_leader:
            wandb.log(
                {
                    "train_loss": loss,
                    "train_accuracy": logs.aux["acc"].mean().item(),
                    "epoch": epoch,
                },
            )


def add_wandb_logger(
    callbacks: List[Callback], opts: argparse.Namespace, game: torch.nn.Module
):
    wandb.init(project="post_rebuttal", tags=[opts.wandb_tag])
    wandb.config.update(opts)
    wandb.watch(game, log="all")

    callbacks.append(WandbLogger())


def get_callbacks(checkpoint_freq: int = 1):
    callbacks = [
        ConsoleLogger(as_json=True, print_train_loss=True),
        BestStatsTracker(),
        VisionModelSaver(checkpoint_freq=checkpoint_freq),
        DistributedSamplerEpochSetter(),
    ]
    return callbacks
