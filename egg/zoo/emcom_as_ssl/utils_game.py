# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torchvision

from egg.zoo.emcom_as_ssl.utils_archs import VisionModule


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


def build_vision_encoder(
    model_name: str = "resnet50",
    shared_vision: bool = False,
    pretrain_vision: bool = False,
):
    (
        sender_vision_module,
        receiver_vision_module,
        visual_features_dim,
    ) = get_vision_modules(
        encoder_arch=model_name,
        shared=shared_vision,
        pretrain_vision=pretrain_vision,
    )
    vision_encoder = VisionModule(
        sender_vision_module=sender_vision_module,
        receiver_vision_module=receiver_vision_module,
    )
    return vision_encoder, visual_features_dim
