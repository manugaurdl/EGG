# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torchvision


def get_cnn(opts):
    modules = {
        "resnet18": torchvision.models.resnet18(pretrained=opts.pretrain_vision),
        "resnet34": torchvision.models.resnet34(pretrained=opts.pretrain_vision),
        "resnet50": torchvision.models.resnet50(pretrained=opts.pretrain_vision),
        "resnet101": torchvision.models.resnet101(pretrained=opts.pretrain_vision),
        "resnet152": torchvision.models.resnet152(pretrained=opts.pretrain_vision),
    }
    if opts.vision_model not in modules:
        raise KeyError(f"{opts.vision_model} is not currently supported.")

    model = modules[opts.vision_model]

    opts.img_feats_dim = model.fc.in_features
    model.fc = nn.Identity()

    if opts.pretrain_vision:
        for param in model.parameters():
            param.requires_grad = False
        model = model.eval()

    return model
