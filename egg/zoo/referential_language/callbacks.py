# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json

import torch

from egg.core import Callback, ConsoleLogger, EarlyStopperAccuracy, Interaction
from egg.core.callbacks import WandbLogger


class VisionModelSaver(Callback):
    """A callback that stores vision module(s) in trainer's checkpoint_dir, if any."""

    def __init__(self, shared_vision: bool, checkpoint_freq: int = 1):
        self.checkpoint_freq = checkpoint_freq
        self.shared = shared_vision

    def on_train_begin(self, trainer_instance: "Trainer"):  # noqa: F821
        self.trainer = trainer_instance

        if self.trainer.distributed_context.is_distributed:
            # if distributed training the model is an instance of
            # DistributedDataParallel and we need to unpack it from it.
            self.game = self.trainer.game.module
        else:
            self.game = self.trainer.game

    def write_model(self, agent_cnn: torch.nn.Module, model_name: str):
        torch.save(
            agent_cnn.state_dict(),
            self.trainer.checkpoint_path / model_name,
        )

    def save_vision_model(self, epoch: str = ""):
        if hasattr(self.trainer, "checkpoint_path") and self.trainer.checkpoint_path:
            self.trainer.checkpoint_path.mkdir(exist_ok=True, parents=True)

            model_name = f"vision_module_{'shared' if self.shared else 'sender'}_{epoch if epoch else 'final'}.pt"
            self.write_model(self.game.sender.agent.vision_module, model_name)

            if not self.shared:
                model_name = f"vision_module_recv_{epoch if epoch else 'final'}.pt"
                self.write_model(self.game.receiver.agent.vision_module, model_name)

    def on_train_end(self):
        if self.trainer.distributed_context.is_leader:
            self.save_vision_model()

    def on_epoch_end(self, loss: float, _logs: Interaction, epoch: int):
        if self.checkpoint_freq > 0 and (epoch % self.checkpoint_freq == 0):
            if self.trainer.distributed_context.is_leader:
                self.save_vision_model(epoch=epoch)


class MyWandbLogger(WandbLogger):
    def on_batch_end(
        self, logs: Interaction, loss: float, batch_id: int, is_training: bool = True
    ):
        if is_training and self.trainer.distributed_context.is_leader:
            self.log_to_wandb(
                {"batch_loss": loss, "batch_accuracy": logs.aux["acc"]}, commit=True
            )

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        if self.trainer.distributed_context.is_leader:
            self.log_to_wandb(
                {"train_loss": loss, "train_accuracy": logs.aux["acc"], "epoch": epoch},
                commit=True,
            )


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


class DistributedSamplerEpochSetter(Callback):
    """A callback that sets the right epoch of a DistributedSampler instance."""

    def on_epoch_begin(self, epoch: int):
        if self.trainer.distributed_context.is_distributed:
            self.trainer.train_data.sampler.set_epoch(epoch)

    def on_validation_begin(self, epoch: int):
        if self.trainer.distributed_context.is_distributed:
            self.trainer.validation_data.sampler.set_epoch(epoch)


def get_callbacks(opts: argparse.Namespace):
    callbacks = [
        BestStatsTracker(),
        ConsoleLogger(as_json=True, print_train_loss=True),
        EarlyStopperAccuracy(0.99, validation=False),
        VisionModelSaver(opts.shared_vision, opts.checkpoint_freq),
    ]

    if opts.distributed_context.is_distributed:
        callbacks.append(DistributedSamplerEpochSetter())

    return callbacks
