# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Union

import torch
import torch.nn as nn
import torchvision
from torch.nn.functional import cosine_similarity as cosine_sim


def initialize_vision_module(name: str = "resnet50", pretrained: bool = False):
    modules = {
        "resnet50": torchvision.models.resnet50(pretrained=pretrained),
        "resnet101": torchvision.models.resnet101(pretrained=pretrained),
        "resnet152": torchvision.models.resnet152(pretrained=pretrained),
    }
    if name not in modules:
        raise KeyError(f"{name} is not currently supported.")

    model = modules[name]

    n_features = model.fc.in_features
    model.fc = nn.Identity()

    if pretrained:
        for param in model.parameters():
            param.requires_grad = False
        model = model.eval()

    return model, n_features


class Sender(nn.Module):
    def __init__(
        self,
        vision_module: Union[nn.Module, str],
        input_dim: Optional[int],
        vocab_size: int = 2048,
    ):
        super(Sender, self).__init__()

        if isinstance(vision_module, nn.Module):
            self.vision_module = vision_module
            input_dim = input_dim
        elif isinstance(vision_module, str):
            self.vision_module, input_dim = initialize_vision_module(vision_module)
        else:
            raise RuntimeError("Unknown vision module for the Sender")

        self.fc_out = nn.Sequential(
            nn.Linear(input_dim, vocab_size, bias=False),
            nn.BatchNorm1d(vocab_size),
        )

    def forward(self, x, aux_input=None):
        [bsz, max_objs, _, h, w] = x.shape
        all_img_feats = self.vision_module(x.view(-1, 3, h, w)).view(bsz * max_objs, -1)
        return self.fc_out(all_img_feats)


class Receiver(nn.Module):
    def __init__(
        self,
        vision_module: Union[nn.Module, str],
        input_dim: int,
        hidden_dim: int = 2048,
        output_dim: int = 2048,
        temperature: float = 1.0,
        cosine_similarity: bool = False,
    ):
        super(Receiver, self).__init__()

        if isinstance(vision_module, nn.Module):
            self.vision_module = vision_module
            input_dim = input_dim
        elif isinstance(vision_module, str):
            self.vision_module, input_dim = initialize_vision_module(vision_module)
        else:
            raise RuntimeError("Unknown vision module for the Receiver")

        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.temperature = temperature
        self.cosine_sim = cosine_sim

    def forward(self, messages, images, aux_input=None):
        [bsz, max_objs, _, h, w] = images.shape
        images = self.vision_module(images.view(-1, 3, h, w))
        aux_input.update({"recv_img_feats": images})
        images = self.fc(images).view(bsz, max_objs, -1)
        messages = messages.view(bsz, max_objs, -1)
        if self.cosine_sim:
            scores = cosine_sim(messages.unsqueeze(2), images.unsqueeze(1), 3)
        else:
            scores = torch.bmm(messages, images.transpose(1, 2))  # dot product sim
        return scores
