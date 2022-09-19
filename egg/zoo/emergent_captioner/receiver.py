# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch.nn as nn

import clip

from egg.zoo.emergent_captioner.utils import convert_models_to_fp32


class ClipReceiver(nn.Module):
    def __init__(
        self,
        clip_model: str,
    ):
        super(ClipReceiver, self).__init__()
        self.clip = clip.load(clip_model)[0]
        convert_models_to_fp32(self.clip)
        self.clip.eval()

    def forward(self, message, images, aux_input=None):
        text = clip.tokenize(message, truncate=True).to(images.device)
        return self.clip.encode_text(text)
