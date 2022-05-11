# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json

import wandb

from egg.core import Callback, Interaction


class WandbLogger(Callback):
    def __init__(
        self,
        opts=None,
        tags=None,
        project=None,
        run_id=None,
        **kwargs,
    ):
        self.opts = opts
        self.last_train_acc = 0.0

        wandb.init(project=project, tags=tags, id=run_id, **kwargs)
        wandb.config.update(opts)

    @staticmethod
    def log_to_wandb(metrics, commit=False, **kwargs):
        wandb.log(metrics, commit=commit, **kwargs)

    def on_train_begin(self, trainer_instance: "Trainer"):  # noqa: F821
        self.trainer = trainer_instance
        wandb.watch(self.trainer.game, log="all")

    def on_batch_end(
        self, logs: Interaction, loss: float, batch_id: int, is_training: bool = True
    ):
        if is_training and self.trainer.distributed_context.is_leader:
            self.log_to_wandb({"batch_loss": loss}, commit=True)
            self.log_to_wandb({"batch_acc": logs.aux["acc"].mean().item()}, commit=True)

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        self.last_train_acc = logs.aux["acc"].mean().item()

        if self.trainer.distributed_context.is_leader:
            self.log_to_wandb({"train_loss": loss, "epoch": epoch}, commit=True)
            self.log_to_wandb(
                {"train_acc": self.last_train_acc, "epoch": epoch}, commit=True
            )

    def on_validation_end(self, loss: float, logs: Interaction, epoch: int):
        if self.trainer.distributed_context.is_leader:
            self.log_to_wandb({"validation_loss": loss, "epoch": epoch}, commit=True)
            self.log_to_wandb(
                {"validation_acc": logs.aux["acc"].mean().item(), "epoch": epoch},
                commit=True,
            )


class BestStatsTracker(Callback):
    def __init__(self):
        super().__init__()
        self.best_val_acc = -float("inf")
        self.best_val_loss = float("inf")
        self.best_val_epoch = -1

    def on_validation_end(self, loss, logs: Interaction, epoch: int):
        if logs.aux["acc"].mean().item() > self.best_val_acc:
            self.best_val_acc = logs.aux["acc"].mean().item()
            self.best_val_epoch = epoch
            self.best_val_loss = loss

    def on_train_end(self):
        valid_stats = dict(
            mode="best validation acc",
            best_epoch=self.best_val_epoch,
            best_acc=self.best_val_acc,
            best_loss=self.best_val_loss,
        )
        print(json.dumps(valid_stats), flush=True)


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
