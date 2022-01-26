# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def get_cnn(name, pretrained):
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


class ScaledDotProductAttention:
    def __call__(self, x):
        return x


class SelfAttention(nn.Module):
    def __init__(self, num_heads):
        super(SelfAttention, self).__init__()

        self.num_heads = num_heads

    def forward(self, x):
        return x


class Attention_topk:
    def __init__(self, k=1):
        self.k = k

    def __call__(self, x):
        bsz, max_objs, _ = x.shape
        mask = torch.ones(max_objs, max_objs, device=x.device).fill_diagonal_(0)
        sims = F.cosine_similarity(x.unsqueeze(2), x.unsqueeze(1), 3)
        sims = sims * mask
        ranks = torch.argsort(sims, descending=True)

        most_similar_dist = []
        for rank in range(self.k):
            top_dist = x[torch.arange(bsz).unsqueeze(-1), ranks[..., rank]]
            most_similar_dist.append(top_dist)

        if self.k == 1:
            most_similar_dist = torch.cat(most_similar_dist, dim=1)
        else:
            # averaging topk distractors with k > 1
            most_similar_dist = torch.stack(most_similar_dist, dim=2).mean(dim=2)
        return torch.cat([x, most_similar_dist], dim=-1)


class Sender(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        attn_fn: Callable,
    ):
        super(Sender, self).__init__()
        self.attn_fn = attn_fn
        self.fc_out = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=False),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(0.02),
        )

    def forward(self, x, aux_input=None):
        bsz, max_objs, _ = x.shape
        x = self.attn_fn(x)
        return self.fc_out(x.view(bsz * max_objs, -1))


class Receiver(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        temperature: int,
    ):
        super(Receiver, self).__init__()
        self.fc_out = nn.Sequential(
            nn.Linear(input_dim, input_dim, bias=False),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim, bias=False),
        )
        self.temperature = temperature

    def forward(self, messages, images, aux_input=None):
        bsz, max_objs, _ = images.shape
        images = self.fc_out(images.view(bsz * max_objs, -1))

        images = images.view(bsz, max_objs, -1)
        messages = messages.view(bsz, max_objs, -1)
        sims = torch.bmm(messages, images.transpose(-1, -2)) / self.temperature
        return sims.view(bsz * max_objs, -1)


class VisionWrapper(nn.Module):
    def __init__(
        self,
        game: nn.Module,
        sender_vision_module: nn.Module,
        recv_vision_module: Optional[nn.Module] = None,
    ):
        super(VisionWrapper, self).__init__()
        self.game = game
        self.encoder_sender = sender_vision_module
        self.shared = recv_vision_module is None
        if not self.shared:
            self.encoder_recv = recv_vision_module

    def forward(self, sender_input, labels, receiver_input=None, aux_input=None):
        bsz, max_objs, _, h, w = sender_input.shape
        sender_input = self.encoder_sender(sender_input.view(bsz * max_objs, 3, h, w))
        receiver_input = sender_input
        if not self.shared:
            receiver_input = self.encoder_recv(
                receiver_input.view(bsz * max_objs, 3, h, w)
            )

        if not self.training:
            aux_input["recv_img_feats"] = receiver_input

        sender_input = sender_input.view(bsz, max_objs, -1)
        receiver_input = receiver_input.view(bsz, max_objs, -1)
        return self.game(sender_input, labels, receiver_input, aux_input)
