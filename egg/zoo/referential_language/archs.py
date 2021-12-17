# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Union

import torch
import torch.nn as nn
import torchvision
from torch.nn.functional import cosine_similarity as cosine_sim

from egg.core.interaction import LoggingStrategy


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


class ContextAttention(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, context_integration: str = "cat"
    ):
        super(ContextAttention, self).__init__()
        self.attn_fn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.context_integration = context_integration

    def forward(self, img_feats, mask):
        context_vectors, attn_weights = self.attn_fn(
            query=img_feats,
            key=img_feats,
            value=img_feats,
            key_padding_mask=mask,
        )
        img_feats = torch.transpose(img_feats, 1, 0)
        context_vectors = torch.transpose(context_vectors, 1, 0)

        if self.context_integration == "cat":
            contextualized_objs = torch.cat([img_feats, context_vectors], dim=-1)
        elif self.context_integration == "gate":
            obj_w_context = img_feats * context_vectors
            context_gate = 1 - torch.sigmoid(obj_w_context)
            contextualized_objs = img_feats * context_gate
        else:
            raise RuntimeError(f"{self.context_integration} not supported")

        return contextualized_objs, attn_weights


class PlusOneWrapper(nn.Module):
    def __init__(self, wrapped):
        super().__init__()
        self.wrapped = wrapped

    def forward(self, *input):
        return self.wrapped(*input) + 1


class Sender(nn.Module):
    def __init__(
        self,
        vision_module: Union[nn.Module, str],
        input_dim: Optional[int],
        output_dim: int = 2048,
        num_heads: int = 0,
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

        assert num_heads >= 0
        self.attention = num_heads > 0
        if self.attention:
            self.attn_fn = ContextAttention(input_dim, num_heads, context_integration)
            input_dim = input_dim * 2 if context_integration == "cat" else input_dim

        self.fc_out = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=False),
            nn.BatchNorm1d(output_dim),
        )

    def forward(self, x, aux_input=None):
        [bsz, max_objs, _, h, w] = x.shape
        img_feats = self.vision_module(x.view(-1, 3, h, w))

        if self.attention:
            # MultiHead attn takes tensor in seq X batch X embedding format
            img_feats = torch.transpose(img_feats.view(bsz, max_objs, -1), 0, 1)
            img_feats, attn_weights = self.attn_fn(img_feats, aux_input["mask"].bool())
            aux_input["attn_weights"] = attn_weights

        return self.fc_out(img_feats.view(bsz * max_objs, -1))


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

    def compute_sim_scores(self, messages, images):
        if self.use_cosine_sim:
            return cosine_sim(messages.unsqueeze(2), images.unsqueeze(1), 3)
        return torch.bmm(messages, images.transpose(1, 2))  # dot product sim

    def forward(self, messages, images, aux_input=None):
        [bsz, max_objs, _, h, w] = images.shape
        images = self.vision_module(images.view(-1, 3, h, w))
        aux_input.update({"recv_img_feats": images})
        images = self.fc(images).view(bsz, max_objs, -1)
        messages = messages.view(bsz, max_objs, -1)
        return self.compute_sim_scores(messages, images)


class SenderReceiverRnnFixedLengthGS(nn.Module):
    def __init__(
        self,
        sender,
        receiver,
        loss,
        length_cost=0.0,
        train_logging_strategy: Optional[LoggingStrategy] = None,
        test_logging_strategy: Optional[LoggingStrategy] = None,
    ):
        super(SenderReceiverRnnFixedLengthGS, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.loss = loss
        self.length_cost = length_cost
        self.train_logging_strategy = (
            LoggingStrategy()
            if train_logging_strategy is None
            else train_logging_strategy
        )
        self.test_logging_strategy = (
            LoggingStrategy()
            if test_logging_strategy is None
            else test_logging_strategy
        )

    def forward(self, sender_input, labels, receiver_input=None, aux_input=None):
        message = self.sender(sender_input, aux_input)
        receiver_output = self.receiver(message, receiver_input, aux_input)

        loss, aux_info = self.loss(
            sender_input,
            message[:, -1, ...],
            receiver_input,
            receiver_output[:, -1, ...],
            labels,
            aux_input,
        )

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )
        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input,
            receiver_input=receiver_input,
            labels=labels,
            aux_input=aux_input,
            receiver_output=receiver_output.detach(),
            message=message.detach(),
            message_length=torch.Tensor([receiver_output.size(1)] * len(sender_input)),
            aux=aux_info,
        )

        return loss.mean(), interaction
