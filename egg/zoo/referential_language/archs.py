# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional

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


class SimpleAttention(nn.Module):
    def forward(self, img_feats, aux_input=None):
        # img_feats = nn.functional.normalize(img_feats, dim=-1)
        img_feats = img_feats / math.sqrt(img_feats.shape[-1])
        sims = torch.bmm(img_feats, img_feats.transpose(1, 2))
        bsz, max_objs = sims.shape[:2]
        self_mask = torch.eye(max_objs, device=img_feats.device)
        self_mask.fill_diagonal_(float("-inf")).repeat(bsz, 1, 1)
        sims += self_mask
        attn = nn.functional.softmax(sims, dim=-1)
        output = torch.bmm(attn, img_feats)
        aux_input["attn_weights"] = attn
        return output


class SelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
    ):
        super(SelfAttention, self).__init__()
        self.attn_fn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, img_feats, aux_input=None):
        # MultiHead attn takes tensor in seq X batch X embedding format
        img_feats = torch.transpose(img_feats, 0, 1)
        attn, attn_weights = self.attn_fn(
            query=img_feats,
            key=img_feats,
            value=img_feats,
            key_padding_mask=aux_input["mask"].bool(),
        )
        ctx_vector = torch.transpose(attn, 1, 0)
        aux_input["attn_weights"] = attn_weights
        return ctx_vector


class Sender(nn.Module):
    def __init__(
        self,
        input_dim: Optional[int],
        output_dim: int = 2048,
        num_heads: int = 0,
        attention_type: str = "none",
        context_integration: str = "cat",
    ):
        super(Sender, self).__init__()
        assert attention_type in ["self", "simple", "none"]
        assert context_integration in ["cat", "gate"]

        self.context_integration = context_integration
        self.attention_type = attention_type
        if self.attention_type == "self":
            self.attn_fn = SelfAttention(input_dim, num_heads)
        elif self.attention_type == "simple":
            self.attn_fn = SimpleAttention()

        if context_integration == "cat" and self.attention_type != "none":
            input_dim *= 2

        self.fc_out = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=False),
            nn.BatchNorm1d(output_dim),
        )

    def _integrate_ctx(self, img_feats, aux_input=None):
        if self.attention_type == "none":
            return img_feats

        context = self.attn_fn(img_feats, aux_input)
        if self.context_integration == "cat":
            # random_idxs = torch.randperm(context.size(2))
            # context = context[:, :, random_idxs]
            contextualized_objs = torch.cat([img_feats, context], dim=-1)
        elif self.context_integration == "gate":
            obj_w_context = img_feats * context
            context_gate = 1 - torch.sigmoid(obj_w_context)
            aux_input["context_gate"] = context_gate
            contextualized_objs = img_feats * context_gate
        return contextualized_objs

    def forward(self, x, aux_input=None):
        [bsz, max_objs, _] = x.shape
        x = self._integrate_ctx(x, aux_input)
        return self.fc_out(x.view(bsz * max_objs, -1))


class Receiver(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 2048,
        output_dim: int = 2048,
        temperature: float = 1.0,
        use_cosine_sim: bool = False,
    ):
        super(Receiver, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
        )
        self.temperature = temperature
        self.use_cosine_sim = use_cosine_sim

    def compute_sim_scores(self, messages, images):
        if self.use_cosine_sim:
            sims = cosine_sim(messages.unsqueeze(2), images.unsqueeze(1), 3)
            return sims / self.temperature
        return torch.bmm(messages, images.transpose(1, 2))  # dot product sim

    def forward(self, messages, images, aux_input=None):
        [bsz, max_objs, _] = images.shape
        messages = messages.view(bsz, max_objs, -1)
        return self.compute_sim_scores(messages, images)


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
        [bsz, max_objs, _, h, w] = sender_input.shape
        sender_input = self.encoder_sender(sender_input.view(-1, 3, h, w))
        sender_input = sender_input.view(bsz, max_objs, -1)
        if self.shared:
            receiver_input = sender_input
        else:
            receiver_input = self.encoder_recv(receiver_input.view(-1, 3, h, w))
            receiver_input = receiver_input.view(bsz, max_objs, -1)
        return self.game(sender_input, labels, receiver_input, aux_input)
