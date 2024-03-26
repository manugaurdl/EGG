import torch
import torch.nn as nn
import torch.nn.functional as F
from egg.zoo.emergent_captioner.finetuning.utils import int2mil, trainable_params
import torch.nn.utils.parametrize as parametrize


class LoRAParametrization(nn.Module):
    def __init__(self, features_in, features_out, rank, alpha, device = None):
        super().__init__()
        # device : same device as feats
        self.lora_A = nn.Parameter(torch.zeros((rank,features_out)).to(device))
        self.lora_B = nn.Parameter(torch.zeros((features_in, rank)).to(device))
        nn.init.normal_(self.lora_A, mean=0, std=1)
        
        # Section 4.1 of the paper: 
        #   We then scale ∆Wx by α/r , where α is a constant in r. 
        #   When optimizing with Adam, tuning α is roughly the same as tuning the learning rate if we scale the initialization appropriately. 
        #   As a result, we simply set α to the first r we try and do not tune it. 
        #   This scaling helps to reduce the need to retune hyperparameters when we vary r.
        self.scale = alpha / rank
        # self.enabled = True

    def forward(self, original_weights):
        # if self.enabled:
        # Return W + (B*A)*scale
        return original_weights + torch.matmul(self.lora_B, self.lora_A).view(original_weights.shape) * self.scale
        # else:
        #     return original_weights


def linear_layer_parameterization(layer, device, rank, lora_alpha=16):
    features_in, features_out = layer.weight.shape
    return LoRAParametrization(
        features_in, features_out, rank=rank, alpha=lora_alpha, device=device
    )

def parameterize(layer, rank):
    parametrize.register_parametrization(
layer, "weight", linear_layer_parameterization(layer, layer.weight.device, rank)
)

def LoRA(model, rank):
    print(f"trainable params before LORA :{trainable_params(model)}")

    for i in range(12):
        parameterize(model.clipcap.gpt.transformer.h[i].attn.c_attn, rank)
        parameterize(model.clipcap.gpt.transformer.h[i].attn.c_proj, rank)
        parameterize(model.clipcap.gpt.transformer.h[i].mlp.c_fc, rank)
        parameterize(model.clipcap.gpt.transformer.h[i].mlp.c_proj, rank)

    #freeze all_params

    
    for name, param in model.named_parameters():
        condition = 'lora' in name or  "gpt.transformer.wte" in name or "clip_project" in name 

        if condition:
            continue
        else:
            param.requires_grad = False

    print(f"trainable params after LORA :{trainable_params(model)}")

