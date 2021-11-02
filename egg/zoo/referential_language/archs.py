# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


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

        self.fc = nn.Sequential(
            nn.Linear(input_dim * 2, vocab_size, bias=False),
            nn.BatchNorm1d(vocab_size),
        )

    @staticmethod
    def dot_product_attention(q, v):
        score = q @ v.t()
        attn = F.softmax(score, dim=-1)
        context = attn @ v
        return context, attn

    def forward(self, x, aux_input=None):
        img_embeddings = self.vision_module(x)

        if img_embeddings.shape[0] == 2:  # we only have to bboxes in the image
            first_target = img_embeddings.view(-1)
            second_target = torch.flip(img_embeddings, dims=[0]).view(-1)
            return self.fc(torch.stack([first_target, second_target], dim=0))

        contexts = []
        for idx, target in enumerate(img_embeddings):
            distractors = torch.cat([img_embeddings[:idx], img_embeddings[idx + 1 :]])
            context, _ = self.dot_product_attention(target.unsqueeze(0), distractors)
            contexts.append(context)
        contexts = torch.cat(contexts)
        output = torch.cat([img_embeddings, contexts], dim=1)
        return self.fc(output)


class Receiver(nn.Module):
    def __init__(
        self,
        vision_module: Union[nn.Module, str],
        input_dim: int,
        hidden_dim: int = 2048,
        output_dim: int = 2048,
        temperature: float = 1.0,
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

    def forward(self, message, distractors, aux_input=None):
        distractors = self.fc(self.vision_module(distractors))

        similarity_scores = (
            torch.nn.functional.cosine_similarity(
                message.unsqueeze(1), distractors.unsqueeze(0), dim=2
            )
            / self.temperature
        )

        return similarity_scores
