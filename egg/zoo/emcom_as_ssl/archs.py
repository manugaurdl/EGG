# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
import torch.nn as nn
import torchvision

from egg.core.continous_communication import SenderReceiverContinuousCommunication
from egg.core.gs_wrappers import gumbel_softmax_sample


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
        encoder_recv = self.encoder if self.shared else self.encoder_recv

        encoded_input_sender, encoded_input_recv = [], []
        for idx in range(x_i.shape[0]):
            encoded_input_sender.append(self.encoder(x_i[idx].unsqueeze(0)))
            encoded_input_recv.append(encoder_recv(x_j[idx].unsqueeze(0)))

        encoded_input_sender = torch.cat(encoded_input_sender).unsqueeze(0).unsqueeze(0)
        encoded_input_recv = torch.cat(encoded_input_recv).unsqueeze(0)

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

    def forward(self, sender_input, labels, receiver_input=None):
        sender_encoded_input, receiver_encoded_input = self.vision_module(
            sender_input, receiver_input
        )

        return self.game(
            sender_input=sender_encoded_input,
            labels=labels,
            receiver_input=receiver_encoded_input,
        )


class SimCLRSender(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 2048,
        output_dim: int = 2048,
        discrete_evaluation: bool = False,
    ):
        super(SimCLRSender, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim, bias=False)

        self.discrete_evaluation = discrete_evaluation

    def forward(self, resnet_output, sender=False):
        first_projection = self.fc(resnet_output)

        if self.discrete_evaluation and (not self.training) and sender:
            logits = first_projection
            size = logits.size()
            indexes = logits.argmax(dim=-1)
            one_hot = torch.zeros_like(logits).view(-1, size[-1])
            one_hot.scatter_(1, indexes.view(-1, 1), 1)
            one_hot = one_hot.view(*size)
            first_projection = one_hot

        out = self.fc_out(first_projection)
        return out, first_projection.detach(), resnet_output.detach()


class EmSSLSender(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 2048,
        output_dim: int = 2048,
        temperature: float = 1.0,
        trainable_temperature: bool = False,
        straight_through: bool = False,
    ):
        super(EmSSLSender, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )

        if not trainable_temperature:
            self.temperature = temperature
        else:
            self.temperature = torch.nn.Parameter(
                torch.tensor([temperature]), requires_grad=True
            )
        self.straight_through = straight_through

        self.fc_out = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, resnet_output):
        first_projection = self.fc(resnet_output)
        message = gumbel_softmax_sample(
            first_projection, self.temperature, self.training, self.straight_through
        )
        out = self.fc_out(message)
        return out, message.detach(), resnet_output.detach()


class InformedSender(nn.Module):
    def __init__(
        self,
        input_dim: int,  # feat_size,
        hidden_dim: int = 20,
        embedding_dim: int = 50,
        vocab_size: int = 2048,
        game_size: int = 2,  # distractors + 1 target)
        temperature: int = 1.0,
    ):
        super(InformedSender, self).__init__()

        self.temperature = temperature

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
        message = gumbel_softmax_sample(h, self.temperature, self.training)
        out = self.fc_out(message)

        return out, message.detach(), x.detach()


class InformedReceiver(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int = 50):
        super(InformedReceiver, self).__init__()
        self.fc = nn.Linear(input_dim, embedding_dim, bias=False)

    def forward(self, _x, resnet_output):
        return self.fc(resnet_output), resnet_output.detach()


class Receiver(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 2048, output_dim: int = 2048):
        super(Receiver, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(2),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, _x, resnet_output):
        return self.fc(resnet_output), resnet_output.detach()


class EmComSSLSymbolGame(SenderReceiverContinuousCommunication):
    def __init__(self, *args, **kwargs):
        super(EmComSSLSymbolGame, self).__init__(*args, **kwargs)

    def forward(self, sender_input, labels, receiver_input=None):
        class_labels, target_position = labels

        if isinstance(self.sender, SimCLRSender):
            message, message_like, resnet_output_sender = self.sender(
                sender_input, sender=True
            )
            receiver_output, _, resnet_output_recv = self.receiver(receiver_input)
        else:
            message, message_like, resnet_output_sender = self.sender(sender_input)
            receiver_output, resnet_output_recv = self.receiver(message, receiver_input)

        loss, aux_info = self.loss(
            sender_input,
            message,
            receiver_input,
            receiver_output,
            labels=target_position,
        )

        aux_info["class_labels"] = class_labels.float()

        if not self.training:
            aux_info["message_like"] = message_like
            aux_info["resnet_output_sender"] = resnet_output_sender
            aux_info["resnet_output_recv"] = resnet_output_recv

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )
        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input,
            receiver_input=receiver_input,
            labels=target_position.float(),
            receiver_output=receiver_output.detach(),
            message=message.detach(),
            message_length=torch.ones(message.size(0)),
            aux=aux_info,
        )

        return loss.mean(), interaction
