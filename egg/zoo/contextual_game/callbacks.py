# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json

from egg.core import Callback, Interaction


class BestStatsTracker(Callback):
    def __init__(self):
        super().__init__()
        self.best_train_acc = -float("inf")
        self.best_train_loss = float("inf")
        self.best_train_epoch = -1

    def on_epoch_end(self, loss, logs: Interaction, epoch: int):
        if logs.aux["acc"].mean().item() > self.best_train_acc:
            self.best_train_acc = logs.aux["acc"].mean().item()
            self.best_train_epoch = epoch
            self.best_train_loss = loss

    def on_train_end(self):
        train_stats = dict(
            mode="best train acc",
            best_epoch=self.best_train_epoch,
            best_acc=self.best_train_acc,
            best_loss=self.best_train_loss,
        )
        print(json.dumps(train_stats), flush=True)


class DistributedSamplerEpochSetter(Callback):
    """A callback that sets the right epoch of a DistributedSampler instance."""

    def __init__(self):
        super().__init__()

    def on_epoch_begin(self, epoch: int):
        if self.trainer.distributed_context.is_distributed:
            self.trainer.train_data.sampler.set_epoch(epoch)

    def on_validation_begin(self, epoch: int):
        if self.trainer.distributed_context.is_distributed:
            self.trainer.validation_data.sampler.set_epoch(epoch)
