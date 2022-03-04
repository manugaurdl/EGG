# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional

import torch
import torch.nn as nn
import torchvision

from egg.core.gs_wrappers import gumbel_softmax_sample
from egg.core.interaction import LoggingStrategy


def get_cnn(opts):
    modules = {
        "resnet18": torchvision.models.resnet18(pretrained=opts.pretrain_vision),
        "resnet34": torchvision.models.resnet34(pretrained=opts.pretrain_vision),
        "resnet50": torchvision.models.resnet50(pretrained=opts.pretrain_vision),
        "resnet101": torchvision.models.resnet101(pretrained=opts.pretrain_vision),
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


class NoAttention(nn.Module):
    def forward(self, x, aux_input=None):
        aux_input["attn_weights"] = None
        return None


class ScaledDotProductAttention(nn.Module):
    def __init__(self, img_feats_dim):
        super(ScaledDotProductAttention, self).__init__()
        self.fc_out = nn.Linear(img_feats_dim, img_feats_dim)

    def forward(self, x, aux_input=None):
        bsz, max_objs, embed_dim = x.shape

        x = x / math.sqrt(embed_dim)
        sims = torch.bmm(x, x.transpose(1, 2))

        non_padded_elems_mask = aux_input["mask"].unsqueeze(1)
        sims = sims.masked_fill(~non_padded_elems_mask, value=float("-inf"))
        self_mask = torch.eye(max_objs, device=x.device).fill_diagonal_(float("-inf"))
        sims += self_mask

        attn_weights = nn.functional.softmax(sims, dim=-1)
        attn = torch.bmm(attn_weights, x)
        attn = self.fc_out(attn)
        if not self.training:
            aux_input["attn_weights"] = attn_weights
        return attn


class SelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super(SelfAttention, self).__init__()
        self.attn_fn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, x, aux_input=None):
        x = x.transpose(0, 1)

        mask = torch.logical_not(aux_input["mask"])
        self_mask = torch.eye(x.shape[0], device=x.device, dtype=torch.bool)

        attn, attn_weights = self.attn_fn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=mask,  # masking padded elements
            attn_mask=self_mask,  # masking self
        )
        attn = attn.transpose(0, 1)
        if not self.training:
            aux_input["attn_weights"] = attn_weights
        return attn


class Attention_topk(nn.Module):
    def __init__(self, img_feats_dim, k=1, random=False):
        super(Attention_topk, self).__init__()
        self.k = k
        self.random = random

        self.fc_out = nn.Linear(img_feats_dim, img_feats_dim)

    def forward(self, x, aux_input=None):
        bsz, max_objs, embed_dim = x.shape
        x = x / math.sqrt(embed_dim)

        # zeroing padded elements so they dont participate in the distractor mean computation
        x = x * aux_input["mask"].unsqueeze(-1).expand_as(x).float()

        sims = torch.bmm(x, x.transpose(1, 2))
        # lower than minimum sim
        sims = sims.masked_fill(~aux_input["mask"].unsqueeze(1), value=-100)
        self_mask = torch.eye(max_objs, device=x.device).fill_diagonal_(float("-inf"))
        sims += self_mask

        ranks = torch.argsort(sims, descending=True)
        if self.random:
            # ranks = ranks[..., torch.randperm(ranks.shape[-1])]
            ranks[..., :-1] = torch.argsort(sims)[..., 1:]

        assert torch.allclose(
            ranks[..., -1], torch.arange(max_objs, device=x.device).repeat(bsz, 1)
        )
        # get topk distractor or all, whatever is smaller if k>0, else all but self if k is negative
        last_k = min(self.k, max_objs - 1) if self.k > 0 else max_objs - 1

        most_similar_dist = []
        for rank in range(last_k):
            top_dist = x[torch.arange(bsz).unsqueeze(-1), ranks[..., rank]]
            most_similar_dist.append(top_dist)

        attn = torch.stack(most_similar_dist, dim=2).sum(dim=2)
        denom = torch.sum(aux_input["mask"].int(), dim=-1) - 1  # excluding self
        attn = attn / torch.clamp(denom, max=last_k).unsqueeze(-1).unsqueeze(-1)

        if not self.training:
            aux_input["attn_weights"] = None
        attn = self.fc_out(attn)
        return attn


class Sender(nn.Module):
    def __init__(
        self,
        attn_fn: nn.Module,
        msg_generator: nn.Module,
    ):
        super(Sender, self).__init__()
        self.attn_fn = attn_fn
        self.msg_generator = msg_generator

    def forward(self, x, aux_input=None):
        attn = self.attn_fn(x, aux_input)
        return self.msg_generator(x, attn, aux_input)


class MessageGeneratorMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        single_symbol,
        temperature,
        separate_mlps,
    ):
        super(MessageGeneratorMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim, bias=False)
        self.fc2 = self.fc1
        if separate_mlps:
            self.fc2 = nn.Linear(input_dim, output_dim, bias=False)
        self.temperature = temperature
        self.single_symbol = single_symbol

    def forward(self, tgt_embedding, ctx_embedding, aux_input=None):
        bsz, max_objs, _ = tgt_embedding.shape

        logits_tgt = self.fc1(tgt_embedding.view(bsz * max_objs, -1))
        msg_tgt = gumbel_softmax_sample(logits_tgt, self.temperature, self.training)
        if self.single_symbol:
            return msg_tgt

        embedding2 = ctx_embedding if ctx_embedding is not None else tgt_embedding
        logits2 = self.fc2(embedding2.contiguous().view(bsz * max_objs, -1))
        msg2 = gumbel_softmax_sample(logits2, self.temperature, self.training)
        msg = torch.stack([msg_tgt, msg2], dim=1)
        return msg


class DoubleSymbolReceiverWrapper(nn.Module):
    def __init__(self, agent, vocab_size, agent_input_size, separate_embeddings: False):
        super(DoubleSymbolReceiverWrapper, self).__init__()
        self.agent = agent

        self.embedding1 = nn.Linear(vocab_size, agent_input_size)
        self.embedding2 = self.embedding1
        if separate_embeddings:
            self.embedding2 = nn.Linear(vocab_size, agent_input_size)

        self.fc_out = nn.Linear(agent_input_size * 2, agent_input_size, bias=False)

    def forward(self, message, input=None, aux_input=None):
        embedded_tgt = self.embedding1(message[:, 0, ...])
        embedded_ctx = self.embedding2(message[:, 1, ...])
        embedded_msg = torch.cat([embedded_tgt, embedded_ctx], dim=1)
        msg = self.fc_out(embedded_msg)
        return self.agent(msg, input, aux_input)


class SingleSymbolReceiverWrapper(nn.Module):
    def __init__(self, agent, vocab_size, agent_input_size, *args, **kwargs):
        super(SingleSymbolReceiverWrapper, self).__init__()
        self.agent = agent
        self.embedding = nn.Linear(vocab_size, agent_input_size)

    def forward(self, message, input=None, aux_input=None):
        embedded_msg = self.embedding(message)
        return self.agent(embedded_msg, input, aux_input)


class Receiver(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        temperature: int,
    ):
        super(Receiver, self).__init__()
        self.fc_img = nn.Sequential(
            nn.Linear(input_dim, input_dim, bias=False),
            nn.Tanh(),
            nn.Linear(input_dim, output_dim, bias=False),
        )
        self.temperature = temperature

    def forward(self, messages, images, aux_input=None):
        bsz, max_objs, _ = images.shape

        images = self.fc_img(images.view(bsz * max_objs, -1))
        images = images.view(bsz, max_objs, -1)
        messages = messages.view(bsz, max_objs, -1)

        sims = torch.bmm(messages, images.transpose(-1, -2)) / self.temperature
        return sims.view(bsz * max_objs, -1)


class VisionWrapper(nn.Module):
    def __init__(
        self,
        game: nn.Module,
        visual_encoder: nn.Module,
        train_logging_strategy: Optional[LoggingStrategy] = None,
        test_logging_strategy: Optional[LoggingStrategy] = None,
    ):
        super(VisionWrapper, self).__init__()
        self.game = game
        self.visual_encoder = visual_encoder

        self.train_logging_strategy = (
            LoggingStrategy().minimal()
            if train_logging_strategy is None
            else train_logging_strategy
        )
        self.test_logging_strategy = (
            LoggingStrategy().minimal()
            if test_logging_strategy is None
            else test_logging_strategy
        )

    def forward(self, sender_input, labels, receiver_input=None, aux_input=None):
        bsz, max_objs, _, h, w = sender_input.shape

        sender_feats = self.visual_encoder(sender_input.view(bsz * max_objs, 3, h, w))
        sender_input = sender_feats.view(bsz, max_objs, -1)

        recv_feats = self.visual_encoder(receiver_input.view(bsz * max_objs, 3, h, w))
        recv_input = recv_feats.view(bsz, max_objs, -1)

        if not self.training:
            aux_input["sender_img_feats"] = sender_input.view(bsz, max_objs, -1)
            aux_input["recv_img_feats"] = recv_input.view(bsz, max_objs, -1)

        loss, interaction = self.game(sender_input, labels, recv_input, aux_input)

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
