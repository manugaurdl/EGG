# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json

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


class DistributedSamplerEpochSetter(Callback):
    def on_epoch_begin(self, epoch: int):
        if self.trainer.distributed_context.is_distributed:
            self.trainer.train_data.sampler.set_epoch(epoch)

    def on_validation_begin(self, epoch: int):
        if self.trainer.distributed_context.is_distributed:
            self.trainer.validation_data.sampler.set_epoch(epoch)


class WandbLogger(Callback):
    def log(self, values):
        if self.trainer.distributed_context.is_leader:
            wandb.log(values)

    def on_batch_end(
        self, logs: Interaction, loss: float, batch_id: int, is_training: bool = True
    ):
        values = {"batch_loss": loss, "batch_acc": logs.aux["acc"].mean()}
        self.log(values)

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        acc = logs.aux["acc"].mean()
        values = {"epoch_loss": loss, "epoch_acc": acc, "epoch": epoch}
        self.log(values)

    def on_validation_end(self, loss: float, logs: Interaction, epoch: int):
        acc = logs.aux["acc"].mean().item()
        values = {"test_loss": loss, "test_acc": acc, "epoch": epoch}
        self.log(values)


def get_callbacks(opts):
    callbacks = [
        BestStatsTracker(),
        ConsoleLogger(as_json=True, print_train_loss=True),
        DistributedSamplerEpochSetter(),
    ]
    if opts.wandb:
        return callbacks + [WandbLogger()]
    return callbacks
