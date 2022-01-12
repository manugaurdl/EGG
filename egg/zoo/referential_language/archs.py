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
        img_feats_norm = nn.functional.normalize(img_feats, dim=-1)
        sims = img_feats_norm @ img_feats_norm.t()
        self_mask = torch.eye(sims.shape[0], device=img_feats.device)
        sims += self_mask.fill_diagonal_(float("-inf"))
        attn_weights = nn.functional.softmax(sims, dim=-1)
        return attn_weights @ img_feats, attn_weights


class SelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, self_mask: bool = False):
        super(SelfAttention, self).__init__()
        self.attn_fn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.self_mask = self_mask

    def forward(self, img_feats):
        # MultiHead attn takes tensor in seq X batch X embedding format
        mask = None
        if self.self_mask:
            mask = torch.eye(img_feats.shape[0], device=img_feats.device).bool()
        attn, attn_weights = self.attn_fn(
            query=img_feats.unsqueeze(1),
            key=img_feats.unsqueeze(1),
            value=img_feats.unsqueeze(1),
            attn_mask=mask,
        )
        return attn.squeeze(), attn_weights.squeeze()


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
        assert attention_type in ["self", "self_mask", "simple", "none"]
        assert context_integration in ["cat", "gate"]

        self.context_integration = context_integration
        self.attention_type = attention_type
        if self.attention_type == "self_mask":
            self.attn_fn = SelfAttention(input_dim, num_heads, self_mask=True)
        elif self.attention_type == "self":
            self.attn_fn = SelfAttention(input_dim, num_heads)
        elif self.attention_type == "simple":
            self.attn_fn = SimpleAttention()

        if self.attention_type != "none" and context_integration == "cat":
            input_dim *= 2

        self.fc_out = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=False),
            nn.BatchNorm1d(output_dim),
        )

    def _integrate_ctx(self, img_feats, aux_input=None):
        if self.attention_type == "none":
            return img_feats

        context, attn_weights = self.attn_fn(img_feats)
        aux_input["attn_weights"] = attn_weights
        if self.context_integration == "cat":
            # random_idxs = torch.randperm(context.size(2))
            # context = context[:, :, random_idxs]
            contextualized_objs = torch.cat([img_feats, context], dim=-1)
        elif self.context_integration == "gate":
            img_feats_norm = nn.functional.normalize(img_feats, dim=-1)
            context_norm = nn.functional.normalize(context, dim=-1)
            obj_w_context = img_feats_norm * (1 - context_norm)
            context_gate = 1 - obj_w_context
            aux_input["context_gate"] = context_norm
            contextualized_objs = img_feats * context_gate
        return contextualized_objs

    def forward(self, x, aux_input=None):
        x = self._integrate_ctx(x, aux_input)
        return self.fc_out(x)


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

    def forward(self, messages, images, aux_input=None):
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

        if self.shared:
            receiver_input = sender_input
            if aux_input is not None:
                aux_input.update({"recv_img_feats": receiver_input})
            else:
                aux_input = {"recv_img_feats": receiver_input}
            return self.game(sender_input, labels, receiver_input, aux_input)

        receiver_input = self.encoder_recv(receiver_input.squeeze())
        if aux_input is not None:
            aux_input.update({"recv_img_feats": receiver_input})
        else:
            aux_input = {"recv_img_feats": receiver_input}
        return self.game(sender_input, labels, receiver_input, aux_input)
