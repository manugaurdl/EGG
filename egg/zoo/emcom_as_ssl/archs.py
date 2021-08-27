# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
import torch.nn as nn
import torchvision


def get_resnet(name: str = "resnet50", pretrained: bool = False):
    """Loads ResNet encoder from torchvision along with features number"""
    resnets = {
        "resnet50": torchvision.models.resnet50(pretrained=pretrained),
        "resnet101": torchvision.models.resnet101(pretrained=pretrained),
        "resnet152": torchvision.models.resnet152(pretrained=pretrained),
    }
    if name not in resnets:
        raise KeyError(f"{name} is not a valid ResNet architecture")

    model = resnets[name]
    n_features = model.fc.in_features
    model.fc = nn.Identity()

    if pretrained:
        for param in model.parameters():
            param.requires_grad = False
        model = model.eval()

    return model, n_features


def get_vision_modules(
    encoder_arch: str, shared: bool = False, pretrain_vision: bool = False
):
    if pretrain_vision:
        assert (
            shared
        ), "A pretrained not shared vision_module is a waste of memory. Please run with --shared set"

    encoder, features_dim = get_resnet(encoder_arch, pretrain_vision)
    encoder_recv = None
    if not shared:
        encoder_recv, _ = get_resnet(encoder_arch)

    return encoder, encoder_recv, features_dim


class VisionModule(nn.Module):
    def __init__(
        self,
        sender_vision_module: nn.Module,
        receiver_vision_module: Optional[nn.Module] = None,
    ):
        super(VisionModule, self).__init__()

        self.encoder = sender_vision_module

        self.shared = receiver_vision_module is None
        if not self.shared:
            self.encoder_recv = receiver_vision_module

    def forward(self, x_i, x_j):
        encoded_input_sender = self.encoder(x_i)
        if self.shared:
            encoded_input_recv = self.encoder(x_j)
        else:
            encoded_input_recv = self.encoder_recv(x_j)
        return encoded_input_sender, encoded_input_recv


class VisionGameWrapper(nn.Module):
    def __init__(
        self,
        game: nn.Module,
        vision_module: nn.Module,
    ):
        super(VisionGameWrapper, self).__init__()
        self.game = game
        self.vision_module = vision_module

    def forward(self, sender_input, labels, receiver_input=None, aux_input=None):
        sender_encoded_input, receiver_encoded_input = self.vision_module(
            sender_input, receiver_input
        )

        return self.game(
            sender_input=sender_encoded_input,
            labels=labels,
            receiver_input=receiver_encoded_input,
            aux_input=aux_input,
        )


class VisionGameWrapperWithInformed(nn.Module):
    def __init__(
        self,
        game: nn.Module,
        vision_module: nn.Module,
    ):
        super(VisionGameWrapperWithInformed, self).__init__()
        self.game = game
        self.vision_module = vision_module

    def forward(self, sender_input, labels, receiver_input=None, aux_input=None):
        sender_encoded_input, receiver_encoded_input = self.vision_module(
            sender_input, receiver_input
        )
        bsz = sender_encoded_input.shape[0]
        sender_encoded_input = sender_encoded_input.view(2, 1, bsz // 2, -1)
        receiver_encoded_input = receiver_encoded_input.view(2, 1, bsz // 2, -1)
        random_order = aux_input["random_order"]
        receiver_encoded_input1 = receiver_encoded_input[0, 0, random_order[0]]
        receiver_encoded_input2 = receiver_encoded_input[1, 0, random_order[1]]
        receiver_encoded_input = torch.stack(
            [receiver_encoded_input1, receiver_encoded_input2]
        )

        return self.game(
            sender_input=sender_encoded_input,
            labels=labels,
            receiver_input=receiver_encoded_input,
            aux_input=aux_input,
        )


class EmSSLSender(nn.Module):
    def __init__(
        self,
        input_dim: int,
        vocab_size: int = 2048,
    ):
        super(EmSSLSender, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, vocab_size),
            nn.BatchNorm1d(vocab_size),
        )

    def forward(self, resnet_output, aux_input=None):
        return self.fc(resnet_output)


class Receiver(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 2048,
        temperature: float = 1.0,
    ):
        super(Receiver, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )
        self.temperature = temperature

    def forward(self, message, resnet_output, aux_input=None):
        distractors = self.fc(resnet_output)
        similarity_scores = (
            torch.nn.functional.cosine_similarity(
                message.unsqueeze(1), distractors.unsqueeze(0), dim=2
            )
            / self.temperature
        )
        return similarity_scores


class InformedSender(nn.Module):
    def __init__(
        self,
        input_dim: int,  # feat_size,
        hidden_dim: int = 20,
        embedding_dim: int = 50,
        vocab_size: int = 2048,
        game_size: int = 2,  # distractors + 1 target)
    ):
        super(InformedSender, self).__init__()

        self.fc_in = nn.Linear(input_dim, embedding_dim, bias=False)
        self.conv1 = nn.Conv2d(
            1,
            hidden_dim,
            kernel_size=(game_size, 1),
            stride=(game_size, 1),
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            1, 1, kernel_size=(hidden_dim, 1), stride=(hidden_dim, 1), bias=False
        )
        self.lin2 = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.fc_out = nn.Linear(vocab_size, embedding_dim, bias=False)

    def forward(self, x, _aux_input=None):
        emb = self.fc_in(x)

        # in: h of size (batch_size, 1, game_size, embedding_size)
        # out: h of size (batch_size, hidden_size, 1, embedding_size)
        h = self.conv1(emb)
        h = torch.sigmoid(h)
        # in: h of size (batch_size, hidden_size, 1, embedding_size)
        # out: h of size (batch_size, 1, hidden_size, embedding_size)
        h = h.transpose(1, 2)
        h = self.conv2(h)
        # h of size (batch_size, 1, 1, embedding_size)
        h = torch.sigmoid(h)
        # h of size (batch_size, embedding_size)
        h = self.lin2(h)
        h = h.squeeze(1).squeeze(1)
        # h of size (batch_size, vocab_size)
        return h


class ReceiverWithInformedSender(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 2048,
        temperature: float = 1.0,
    ):
        super(ReceiverWithInformedSender, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.temperature = temperature

    def forward(self, message, resnet_output, aux_input=None):
        distractors = self.fc(resnet_output)
        similarity_scores = (
            torch.nn.functional.cosine_similarity(
                message.unsqueeze(1), distractors, dim=2
            )
            / self.temperature
        )
        return similarity_scores
