# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import uuid

import torch

from egg.core import Callback, ConsoleLogger, EarlyStopperAccuracy, Interaction
from egg.core.callbacks import WandbLogger


class VisionModelSaver(Callback):
    """A callback that stores vision module(s) in trainer's checkpoint_dir, if any."""

    def save_vision_model(self, epoch=""):
        if hasattr(self.trainer, "checkpoint_path"):
            self.trainer.checkpoint_path.mkdir(exist_ok=True, parents=True)
            vision_module = self.trainer.game.sender.vision_encoder

            model_name = f"vision_module_{epoch if epoch else 'final'}.pt"
            torch.save(
                vision_module.state_dict(),
                self.trainer.checkpoint_path / model_name,
            )

    def on_train_end(self):
        self.save_vision_model()

    def on_epoch_end(self, loss: float, _logs: Interaction, epoch: int):
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
        super().__init__()

        # TRAIN
        self.best_train_acc, self.best_train_loss, self.best_train_epoch = (
            -float("inf"),
            float("inf"),
            -1,
        )
        self.last_train_acc, self.last_train_loss, self.last_train_epoch = 0.0, 0.0, 0
        # last_val_epoch useful for runs that end before the final epoch

        self.best_val_acc, self.best_val_loss, self.best_val_epoch = (
            -float("inf"),
            float("inf"),
            -1,
        )
        self.last_val_acc, self.last_val_loss, self.last_val_epoch = 0.0, 0.0, 0
        # last_{train, val}_epoch useful for runs that end before the final epoch

    def on_epoch_end(self, loss, logs: Interaction, epoch: int):
        if logs.aux["acc"].mean().item() > self.best_train_acc:
            self.best_train_acc = logs.aux["acc"].mean().item()
            self.best_train_epoch = epoch
            self.best_train_loss = loss

        self.last_train_acc = logs.aux["acc"].mean().item()
        self.last_train_epoch = epoch
        self.last_train_loss = loss

    def on_validation_end(self, loss, logs: Interaction, epoch: int):
        if logs.aux["acc"].mean().item() > self.best_val_acc:
            self.best_val_acc = logs.aux["acc"].mean().item()
            self.best_val_epoch = epoch
            self.best_val_loss = loss

        self.last_val_acc = logs.aux["acc"].mean().item()
        self.last_val_epoch = epoch
        self.last_val_loss = loss

    def on_train_end(self):
        train_stats = dict(
            mode="best train acc",
            best_epoch=self.best_train_epoch,
            best_acc=self.best_train_acc,
            best_loss=self.best_train_loss,
            last_epoch=self.last_train_epoch,
            last_acc=self.last_train_acc,
            last_loss=self.last_train_loss,
        )
        print(json.dumps(train_stats), flush=True)
        val_stats = dict(
            mode="best validation acc",
            best_epoch=self.best_val_epoch,
            best_acc=self.best_val_acc,
            best_loss=self.best_val_loss,
            last_epoch=self.last_val_epoch,
            last_acc=self.last_val_acc,
            last_loss=self.last_val_loss,
        )
        print(json.dumps(val_stats), flush=True)


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


def get_callbacks(opts):
    callbacks = [
        BestStatsTracker(),
        ConsoleLogger(as_json=True, print_train_loss=True),
        EarlyStopperAccuracy(0.99, validation=False),
        VisionModelSaver(),
    ]

    if opts.wandb:
        run_name = opts.checkpoint_dir.split("/")[-1] if opts.checkpoint_dir else ""
        callbacks.append(
            MyWandbLogger(
                opts, "contezualized_emcomm", f"{run_name}_{str(uuid.uuid4())}"
            )
        )

    if opts.distributed_contex.is_distributed:
        callbacks.append(DistributedSamplerEpochSetter())

    return callbacks
