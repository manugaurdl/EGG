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
        attention: bool = False,
        num_heads: int = 1,
        context_integration: str = "cat",
    ):
        super(Sender, self).__init__()

        if isinstance(vision_module, nn.Module):
            self.vision_module = vision_module
            input_dim = input_dim
        elif isinstance(vision_module, str):
            self.vision_module, input_dim = initialize_vision_module(vision_module)
        else:
            raise RuntimeError("Unknown vision module for the Sender")

        fc_out_dim = input_dim
        self.attention = attention
        if self.attention:
            self.attn_fn = nn.MultiheadAttention(
                embed_dim=input_dim, num_heads=num_heads
            )
            fc_out_dim = input_dim * 2 if context_integration == "cat" else fc_out_dim
        self.context_integration = context_integration

        self.fc_out = nn.Sequential(
            nn.Linear(fc_out_dim, vocab_size, bias=False),
            nn.BatchNorm1d(vocab_size),
        )

    def expand_mask(self, mask: torch.Tensor, max_objs: int) -> torch.Tensor:
        """Expand a 1D with num of padded objs into a boolean 2D mask of size bsz X max_objs."""
        bsz = mask.shape[0]
        # ones mean masking for the purpose of (self) attention
        expanded_mask = torch.ones(bsz, max_objs, device=mask.device)
        for idx, num_elems in enumerate(mask):
            if num_elems > 0:
                not_padded_idx = int(max_objs - num_elems)
                expanded_mask[idx][:not_padded_idx] = 0
        return expanded_mask.bool()

    def forward(self, x, aux_input=None):
        [bsz, max_objs, _, h, w] = x.shape
        all_img_feats = self.vision_module(x.view(-1, 3, h, w))

        if self.attention:
            expanded_mask = self.expand_mask(aux_input["mask"], max_objs)
            all_img_feats = torch.transpose(all_img_feats.view(bsz, max_objs, -1), 0, 1)
            context_vectors, attn_weights = self.attn_fn(
                query=all_img_feats,
                key=all_img_feats,
                value=all_img_feats,
                key_padding_mask=expanded_mask,
            )
            all_img_feats = torch.transpose(all_img_feats, 1, 0)
            context_vectors = torch.transpose(context_vectors, 1, 0)
            if aux_input:
                aux_input["attn_weights"] = attn_weights
            else:
                aux_input = {"attn_weights": attn_weights}
            if self.context_integration == "cat":
                all_img_feats = torch.cat([all_img_feats, context_vectors], dim=-1)
            elif self.context_integration == "gate":
                obj_w_context = all_img_feats * context_vectors
                context_gate = 1 - torch.sigmoid(obj_w_context)
                all_img_feats = all_img_feats * context_gate
            else:
                raise RuntimeError(f"{self.context_integration} not supported")
        out = self.fc_out(all_img_feats.view(bsz * max_objs, -1))
        return out


class Receiver(nn.Module):
    def __init__(
        self,
        vision_module: Union[nn.Module, str],
        input_dim: int,
        hidden_dim: int = 2048,
        output_dim: int = 2048,
        temperature: float = 1.0,
        use_cosine_sim: bool = False,
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
        self.use_cosine_sim = use_cosine_sim

    def forward(self, messages, images, aux_input=None):
        [bsz, max_objs, _, h, w] = images.shape
        images = self.vision_module(images.view(-1, 3, h, w))
        aux_input.update({"recv_img_feats": images})
        images = self.fc(images).view(bsz, max_objs, -1)
        messages = messages.view(bsz, max_objs, -1)
        if self.use_cosine_sim:
            scores = cosine_sim(messages.unsqueeze(2), images.unsqueeze(1), 3)
        else:
            scores = torch.bmm(messages, images.transpose(1, 2))  # dot product sim
        return scores
