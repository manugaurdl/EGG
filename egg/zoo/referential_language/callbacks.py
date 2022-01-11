# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json

import wandb

from egg.core import Callback, ConsoleLogger, Interaction
from egg.core.early_stopping import EarlyStopper


class BestStatsTracker(Callback):
    def __init__(self):
        self.best = {"acc": -float("inf"), "loss": float("inf"), "epoch": -1}

    def on_validation_end(self, loss, logs: Interaction, epoch: int):
        if logs.aux["acc"].mean().item() > self.best["acc"]:
            self.best["acc"] = logs.aux["acc"].mean().item()
            self.best["loss"] = loss
            self.best["epoch"] = epoch

    def on_train_end(self):
        best_stats = dict(mode="best_stats", **self.best)
        print(json.dumps(best_stats), flush=True)


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


class EarlyStopperAcc(EarlyStopper):
    def should_stop(self) -> bool:
        if len(self.validation_stats) > 2:
            past_acc = self.validation_stats[-2][1].aux["acc"].mean().item()
            last = self.validation_stats[-1][1].aux["acc"].mean().item()
            if (last - past_acc.item()) * 100 > 0.001:
                return False
            return True


def get_callbacks(opts):
    callbacks = [
        BestStatsTracker(),
        ConsoleLogger(as_json=True, print_train_loss=True),
        EarlyStopperAcc(),
    ]
    if opts.wandb:
        return callbacks + [WandbLogger()]
    return callbacks
