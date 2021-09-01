# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch.nn as nn


class VisionModule(nn.Module):
    def __init__(
        self,
        sender_vision_module: nn.Module,
        receiver_vision_module: Optional[nn.Module] = None,
    ):
        super(VisionModule, self).__init__()

        self.encoder = sender_vision_module

        self.shared = receiver_vision_module is None
        if not self.shared:
            self.encoder_recv = receiver_vision_module

    def forward(self, x_i, x_j):
        encoded_input_sender = self.encoder(x_i)
        if self.shared:
            encoded_input_recv = self.encoder(x_j)
        else:
            encoded_input_recv = self.encoder_recv(x_j)
        return encoded_input_sender, encoded_input_recv
