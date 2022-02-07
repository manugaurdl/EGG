# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from egg.core import Callback, ConsoleLogger
from egg.core.early_stopping import EarlyStopper


class EarlyStopperAcc(EarlyStopper):
    def should_stop(self) -> bool:
        if len(self.validation_stats) > 1:
            past_acc = self.validation_stats[-2][1].aux["acc"].mean().item()
            last = self.validation_stats[-1][1].aux["acc"].mean().item()
            if (last - past_acc) * 100 > 0.001:
                return False
            print("| EARLY STOPPING")
            return True

        return False


class DistributedSamplerEpochSetter(Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_begin(self, epoch: int):
        if self.trainer.distributed_context.is_distributed:
            self.trainer.train_data.sampler.set_epoch(epoch)

    def on_validation_begin(self, epoch: int):
        if self.trainer.distributed_context.is_distributed:
            self.trainer.validation_data.sampler.set_epoch(epoch)


def get_callbacks(opts):
    return [
        ConsoleLogger(as_json=True, print_train_loss=True),
        EarlyStopperAcc(),
        DistributedSamplerEpochSetter(),
    ]
