# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

import torch.nn.functional as F
from egg.core import Callback, ConsoleLogger, Interaction
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


class InteractionPadder(Callback):
    def __init__(self, max_objs, random_distractors):
        self.max_objs = max_objs
        self.random_distractors = random_distractors

    def on_batch_end(
        self, logs: Interaction, loss: float, batch_id: int, is_training: bool = True
    ):
        pad_len = self.max_objs - logs.aux["acc"].shape[0]
        logs.aux["baseline"] = torch.Tensor([1 / logs.aux["acc"].shape[0]])

        if not is_training:
            if not logs.aux_input:
                logs.aux_input = {}
            for k, v in logs.aux_input.items():
                if k == "attn_weights":
                    logs.aux_input[k] = F.pad(v, (0, pad_len, 0, pad_len))
                else:
                    logs.aux_input[k] = F.pad(v, (0, 0, 0, pad_len))

            logs.aux_input["single_accs"] = F.pad(
                logs.aux["acc"], (0, pad_len), value=-1
            )

            if logs.receiver_input is not None:  # test set, not validation
                logs.aux_input["mask"] = torch.ones(self.max_objs)
                if pad_len > 0:
                    logs.aux_input["mask"][-pad_len:] = 0
                logs.receiver_input = F.pad(logs.receiver_input, (0, 0, 0, pad_len))
            if logs.labels is not None:
                logs.labels = F.pad(logs.labels, (0, pad_len), value=-1)
            if logs.message is not None:
                logs.message = F.pad(logs.message, (0, 0, 0, pad_len), value=-1)
            if logs.receiver_output is not None:
                logs.receiver_output = F.pad(
                    logs.receiver_output, (0, pad_len, 0, pad_len), value=-1
                )

        if self.random_distractors and not is_training:
            acc = logs.aux["acc"][0].item()
        else:
            acc = logs.aux["acc"].mean().item()
        logs.aux["acc"] = torch.Tensor([acc])


def get_callbacks(opts):
    callbacks = [
        ConsoleLogger(as_json=True, print_train_loss=True),
        EarlyStopperAcc(),
        InteractionPadder(opts.max_objects, opts.random_distractors),
    ]
    return callbacks
