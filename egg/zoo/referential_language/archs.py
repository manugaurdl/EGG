# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
import torch.nn as nn
import torchvision
from torch.nn.functional import cosine_similarity as cosine_sim


def get_cnn(name: str = "resnet50", pretrained: bool = False):
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
        sims = img_feats @ img_feats.t()
        self_mask = torch.eye(sims.shape[0], device=img_feats.device)
        sims += self_mask.fill_diagonal_(float("-inf"))
        attn_weights = nn.functional.softmax(sims, dim=-1)

        """
        random_idxs = torch.randperm(attn_weights.size(1))
        attn_weights = attn_weights[:, random_idxs]
        attn_weights = torch.zeros_like(attn_weights)
        random_idxs = torch.randint(0, len(attn_weights), (len(attn_weights),))
        attn_weights[torch.arange(len(attn_weights)), random_idxs] = 1
        assert torch.sum(attn_weights).item() == len(attn_weights)
        """
        return attn_weights @ img_feats, attn_weights


class SimpleLinearAttention(SimpleAttention):
    def __init__(self, input_dim):
        super(SimpleLinearAttention, self).__init__()
        self.fc_in = nn.Sequential(nn.Linear(input_dim, input_dim), nn.Tanh())

    def forward(self, img_feats, aux_input=None):
        img_feats = self.fc_in(img_feats)
        return super().forward(img_feats, aux_input)


class Sender(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 2048,
        attention_type: str = "none",
        context_integration: str = "cat",
    ):
        super(Sender, self).__init__()
        assert attention_type in [
            "simple",
            "simple_linear",
            "none",
        ]
        assert context_integration in ["cat", "gate"]

        self.fc_in = nn.Sequential(nn.Linear(input_dim, input_dim), nn.Tanh())
        self.context_integration = context_integration
        self.attention_type = attention_type

        if self.attention_type == "simple":
            self.attn_fn = SimpleAttention()
        elif self.attention_type == "simple_linear":
            self.attn_fn = SimpleLinearAttention(input_dim)

        if self.attention_type != "none":
            if context_integration == "cat":
                input_dim *= 2
            elif context_integration == "gate":
                self.fc_ctx = nn.Sequential(
                    nn.Linear(input_dim, input_dim), nn.Sigmoid()
                )

        self.fc_out = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=False),
            nn.BatchNorm1d(output_dim),
        )

    def _integrate_ctx(self, img_feats, aux_input=None):
        if self.attention_type == "none":
            return img_feats

        context, attn_weights = self.attn_fn(img_feats)
        if not self.training:
            aux_input["attn_weights"] = attn_weights
        if self.context_integration == "cat":
            # random_idxs = torch.randperm(context.size(1))
            # context = context[:, random_idxs]
            contextualized_objs = torch.cat([img_feats, context], dim=-1)
        elif self.context_integration == "gate":
            context = self.fc_ctx(context)
            context_gate = img_feats * context
            if not self.training:
                aux_input["context_gate"] = context_gate
            contextualized_objs = img_feats * context_gate
        return contextualized_objs

    def forward(self, x, aux_input=None):
        x = self.fc_in(x)
        x = self._integrate_ctx(x, aux_input)
        return self.fc_out(x)


class Receiver(nn.Module):
    def __init__(
        self,
        input_dim: int,
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

    def forward(self, messages, images, aux_input=None):
        images = self.fc(images)
        if self.use_cosine_sim:
            sims = cosine_sim(messages.unsqueeze(1), images.unsqueeze(0), 2)
            return sims / self.temperature
        return messages @ images.t()


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
        labels = labels.squeeze()
        sender_input = self.encoder_sender(sender_input.squeeze())
        receiver_input = sender_input

        if not self.shared:
            receiver_input = self.encoder_recv(receiver_input.squeeze())
        if not self.training:
            aux_input = {"recv_img_feats": receiver_input}
        return self.game(sender_input, labels, receiver_input, aux_input)
