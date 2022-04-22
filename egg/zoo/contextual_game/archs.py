# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import clip
import torch.nn as nn
import torchvision
from torch.nn.functional import cosine_similarity as cosine_sim

from egg.core.interaction import LoggingStrategy


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()


def initialize_vision_module(name: str = "resnet50", pretrained: bool = False):
    modules = {
        "resnet50": torchvision.models.resnet50(pretrained=pretrained),
        "resnet101": torchvision.models.resnet101(pretrained=pretrained),
        "resnet152": torchvision.models.resnet152(pretrained=pretrained),
        "vgg11": torchvision.models.vgg11(pretrained=pretrained),
        "clip_vit_b/32": clip.load("ViT-B/32")[0].visual,
        "clip_vit_b/16": clip.load("ViT-B/16")[0].visual,
        "clip_vit_l/14": clip.load("ViT-L/14")[0].visual,
        "clip_resnet50": clip.load("RN50")[0].visual,
        "clip_resnet101": clip.load("RN101")[0].visual,
    }

    if name not in modules:
        raise KeyError(f"{name} is not currently supported.")

    model = modules[name]

    if name in ["resnet50", "resnet101", "resnet152"]:
        n_features = model.fc.in_features
        model.fc = nn.Identity()

    elif name == "vgg11":
        n_features = model.classifier[6].in_features
        model.classifier[6] = nn.Identity()
    else:  # clip
        n_features = model.output_dim
        convert_models_to_fp32(model)

    if pretrained:
        for param in model.parameters():
            param.requires_grad = False
        model = model.eval()

    return model, n_features


class Sender(nn.Module):
    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        temperature: float = 1.0,
    ):
        super(Sender, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, vocab_size),
            nn.BatchNorm1d(vocab_size),
        )

        self.temperature = temperature

    def forward(self, resnet_output, aux_input=None):
        return self.fc(resnet_output)


class Receiver(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 2048,
        output_dim: int = 2048,
        temperature: float = 1.0,
    ):
        super(Receiver, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )
        self.temperature = temperature

    def forward(self, message, candidates, aux_input=None):
        candidates = self.fc(candidates)
        sims = cosine_sim(message.unsqueeze(1), candidates.unsqueeze(0), dim=2)
        return sims / self.temperature


class VisionWrapper(nn.Module):
    def __init__(
        self,
        game: nn.Module,
        visual_encoder: nn.Module,
    ):
        super(VisionWrapper, self).__init__()
        self.game = game
        self.visual_encoder = visual_encoder

        self.train_logging_strategy = LoggingStrategy().minimal()
        self.test_logging_strategy = LoggingStrategy()

    def forward(self, sender_input, labels, receiver_input=None, aux_input=None):
        # TODO: currently there's no support for non-shared vision modules
        visual_feats = self.visual_encoder(sender_input)

        loss, interaction = self.game(visual_feats, labels, visual_feats, aux_input)

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )

        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input,
            receiver_input=receiver_input,
            labels=labels,
            aux_input=interaction.aux_input,
            receiver_output=interaction.receiver_output,
            message=interaction.message,
            message_length=interaction.message_length,
            aux=interaction.aux,
        )
        return loss, interaction
