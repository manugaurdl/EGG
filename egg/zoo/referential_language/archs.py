# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as vision_F

from egg.core.continous_communication import SenderReceiverContinuousCommunication


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


def dot_product_attention(q, v):
    score = q @ v.t()
    attn = F.softmax(score, dim=-1)
    context = attn @ v
    return context, attn


"""
contexts = []
for idx, target in enumerate(img_embeddings):
    distractors = torch.cat([img_embeddings[:idx], img_embeddings[idx + 1 :]])
    context, _ = self.dot_product_attention(target.unsqueeze(0), distractors)
    contexts.append(context)
contexts = torch.cat(contexts)
output = torch.cat([img_embeddings, contexts], dim=1)
"""


class Cropper(nn.Module):
    def forward(self, imgs: torch.Tensor, bboxes: Sequence[int]):
        final = []
        for img, img_bboxes in zip(imgs, bboxes):
            objs = []
            for bbox in img_bboxes:
                top, left = int(bbox[2]), int(bbox[0])
                h = max(1, int(bbox[3] - bbox[2]))
                w = max(1, int(bbox[1] - bbox[0]))
                crop = vision_F.crop(img, top=top, left=left, height=h, width=w)
                objs.append(crop)
            final.append(torch.stack(objs))
        return final


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

    def forward(self, all_imgs: List[torch.Tensor], aux_input: Dict[Any] = None):
        # input: list of tensors each of size n_obj X 3 X H X W
        # output: list of tensor each of size n_obj X input_dim
        img_embeddings = [self.vision_module(objs) for objs in all_imgs]

        return self.fc_out(output)


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


class ContextualCommunicationGame(SenderReceiverContinuousCommunication):
    def __init__(self):
        pass
