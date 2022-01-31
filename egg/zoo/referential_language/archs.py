# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Callable, Optional

import torch
import torch.nn as nn

# import torch.nn.functional as F
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
    def __call__(self, x, aux_input=None):
        bsz, max_objs, embed_dim = x.shape
        x = x / math.sqrt(embed_dim)

        sims = torch.bmm(x, x.transpose(1, 2))

        padded_elems_mask = ~aux_input["mask"].unsqueeze(-2).expand_as(sims)
        sims = sims.masked_fill(padded_elems_mask, value=float("-inf"))
        self_mask = torch.eye(max_objs, device=x.device).fill_diagonal_(float("-inf"))
        sims += self_mask

        attn_weights = nn.functional.softmax(sims, dim=-1)
        attn = torch.bmm(attn_weights, x)
        return attn, attn_weights


class SelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super(SelfAttention, self).__init__()
        self.attn_fn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, x, aux_input=None):
        x = x.transpose(0, 1)

        mask = torch.logical_not(aux_input["mask"])
        self_mask = torch.eye(x.shape[0], device=x.device, dtype=torch.bool)

        attn, attn_weights = self.attn_fn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=mask,  # masking padded elements
            attn_mask=self_mask,  # masking self
        )
        attn = attn.transpose(0, 1)

        return attn, attn_weights


class Attention_topk:
    def __init__(self, k=1, random=False):
        self.k = k
        self.random = random

    def __call__(self, x, aux_input=None):
        bsz, max_objs, embed_dim = x.shape
        x = x / math.sqrt(embed_dim)

        # zeroing padded elements so they dont participate in the distractor mean computation
        x = x * aux_input["mask"].unsqueeze(-1).expand_as(x).float()

        sims = torch.bmm(x, x.transpose(1, 2))
        padded_elems_mask = ~aux_input["mask"].unsqueeze(-2).expand_as(sims)
        sims = sims.masked_fill(padded_elems_mask, value=2)  # lower than minimum sim
        self_mask = torch.eye(max_objs, device=x.device).fill_diagonal_(float("-inf"))
        sims += self_mask

        ranks = torch.argsort(sims, descending=True)
        assert torch.allclose(
            ranks[..., -1], torch.arange(max_objs, device=x.device).repeat(bsz, 1)
        )
        if self.random:
            ranks = ranks[..., torch.randperm(ranks.shape[-1])]

        # get topk distractor or all, whatever is smaller if k>0, else all but self if k is negative
        last_k = min(self.k, max_objs - 1) if self.k > 0 else max_objs - 1

        most_similar_dist = []
        for rank in range(last_k):
            top_dist = x[torch.arange(bsz).unsqueeze(-1), ranks[..., rank]]
            most_similar_dist.append(top_dist)

        attn = torch.stack(most_similar_dist, dim=2).sum(dim=2)
        denom = torch.sum(aux_input["mask"].int(), dim=-1) - 1  # excluding self
        attn = attn / torch.clamp(denom, max=last_k).unsqueeze(-1).unsqueeze(-1)

        return attn, torch.zeros(1, device=x.device)  # dummy attn_weights


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
        attn, attn_weights = self.attn_fn(x, aux_input)
        aux_input["attn_weights"] = attn_weights

        # TODO: this cat should be moved inside the attention functions
        x = torch.cat([x, attn], dim=-1)

        if "global_context_feats" in aux_input:
            x = torch.cat([x, aux_input["global_context_feats"]], dim=-1)

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
        global_context: bool = True,
    ):
        super(VisionWrapper, self).__init__()
        self.game = game
        self.encoder_sender = sender_vision_module
        self.shared = recv_vision_module is None
        if not self.shared:
            self.encoder_recv = recv_vision_module

        self.global_context = global_context

    def forward(self, sender_input, labels, receiver_input=None, aux_input=None):
        bsz, max_objs, _, h, w = sender_input.shape

        if "mask" not in aux_input:
            mask = torch.ones(bsz, max_objs, device=sender_input.device)
            aux_input["mask"] = mask.bool()

        if self.global_context:
            ctx = self.encoder_sender(aux_input["global_context"])
            aux_input["global_context_feats"] = ctx.unsqueeze(1).repeat(1, max_objs, 1)

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
