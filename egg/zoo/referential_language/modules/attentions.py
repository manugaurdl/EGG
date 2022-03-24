# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, embed_dim, **kwargs):
        super(ScaledDotProductAttention, self).__init__()
        self.fc_out = nn.Linear(embed_dim, embed_dim)

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


class MHAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, **kwargs):
        super(MHAttention, self).__init__()
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


class AttentionTopK(nn.Module):
    def __init__(self, embed_dim, attn_topk=1, random_k=False, **kwargs):
        super(AttentionTopK, self).__init__()
        self.k = attn_topk
        self.random = random_k

        self.fc_out = nn.Linear(embed_dim, embed_dim)

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


class TargetAttention(nn.Module):
    def __init__(self, embed_dim, **kwargs):
        super(TargetAttention, self).__init__()

    def forward(self, x, aux_input=None):
        return x


class NoAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        super(NoAttention, self).__init__()

    def forward(self, x, aux_input=None):
        return torch.Tensor([]).to(x.device)


class RandomContextAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        super(RandomContextAttention, self).__init__()

    def forward(self, x, aux_input=None):
        bsz, max_objs = x.shape[:2]

        random_idxs = (torch.arange(bsz) + 1) % bsz

        ctx = x.clone().detach()
        ctx[torch.arange(bsz)] = ctx[random_idxs]

        return ctx


class RandomAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        super(RandomAttention, self).__init__()

    def forward(self, x, aux_input=None):
        return torch.rand_like(x, device=x.device).detach()
