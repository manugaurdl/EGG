# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json

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


def get_callbacks():
    return [BestStatsTracker(), ConsoleLogger(as_json=True, print_train_loss=True)]
