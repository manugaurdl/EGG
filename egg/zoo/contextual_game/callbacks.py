# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import wandb

from egg.core import Callback, Interaction


class WandbLogger(Callback):
    def __init__(
        self,
        opts=None,
        tags=None,
        project=None,
        **kwargs,
    ):
        wandb.init(project=project, tags=tags, **kwargs)
        wandb.config.update(opts)

    @staticmethod
    def log_to_wandb(metrics, commit=False, **kwargs):
        wandb.log(metrics, commit=commit, **kwargs)

    def on_batch_end(
        self, logs: Interaction, loss: float, batch_id: int, is_training: bool = True
    ):
        if self.trainer.distributed_context.is_leader:
            self.log_to_wandb({"batch_acc": logs.aux["acc"].mean().item()}, commit=True)

    def on_validation_end(self, loss: float, logs: Interaction, epoch: int):
        if self.trainer.distributed_context.is_leader:
            acc = logs.aux["acc"].mean().item()
            self.log_to_wandb({"validation_acc": acc, "epoch": epoch}, commit=True)
