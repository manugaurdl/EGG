# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F

from egg.zoo.emergent_captioner.utils import DATASET2NEG_PATHS


class Loss(nn.Module):
    def __init__(
        self,
        train_emb_path: str = None,
        train_nns_path: str = None,
        num_hard_negatives: int = 0,
    ):
        super().__init__()

        self.train_emb = None
        self.train_nns = None
        train_emb_path = None
        if train_emb_path:
            assert train_nns_path
            self.emb = torch.load(train_emb_path, map_location="cpu")
            self.nns = torch.load(train_nns_path, map_location="cpu")

        self.num_hard_negatives = num_hard_negatives

    def get_similarity_scores(self, text_feats, image_feats, img_idxs, aux_input=None):
        cosine_in_batch = text_feats @ image_feats.t()

        targets = cosine_in_batch.diag(0).unsqueeze(1)
        cosine_in_batch.fill_diagonal_(float("-inf"))
        cosine_in_batch = torch.cat([targets, cosine_in_batch], dim=1)

        cosine_sims = cosine_in_batch

        if self.num_hard_negatives > 0 and self.nns:
            elem_idxs = img_idxs.squeeze()

            # fetches embeddings of nearest-neighbor hard negatives
            batch_nns = self.nns[elem_idxs][:, 1 : self.num_hard_negatives + 1].long()

            # batch x num_negatives x embed_dim
            image_feats_negatives = self.emb[batch_nns].to(text_feats.device)[1:]

            cosine_negatives = torch.einsum(
                "be,bne->bn", text_feats, image_feats_negatives
            )

            cosine_sims = torch.cat([cosine_in_batch, cosine_negatives], dim=1)

        aux_input["receiver_output"] = cosine_sims.detach()

        return cosine_sims

    def remove_fields_negatives(self):
        self.nns = None
        self.emb = None

    def forward(self, text_feats, img_feats, img_idxs, aux_input=None):
        raise NotImplementedError


class DiscriminativeLoss(Loss):
    def forward(self, text_feats, img_feats, img_idxs, aux_input=None):
        sims = self.get_similarity_scores(text_feats, img_feats, img_idxs, aux_input) # (B, 3)

        labels = torch.zeros(sims.shape[0]).long().to(img_feats.device)
        loss = F.cross_entropy(sims, labels, reduction="none")
        acc = (sims.argmax(dim=1) == labels).detach().float() # highest similarity index  == labels (indices)

        return loss, {"acc": acc}


class AccuracyLoss(Loss):
    def forward(self, text_feats, img_feats, img_idxs, aux_input=None):
        sims = self.get_similarity_scores(text_feats, img_feats, img_idxs, aux_input)

        labels = torch.zeros(sims.shape[0]).long().to(img_feats.device)

        acc = (sims.argmax(dim=1) == labels).detach().float()

        return -acc, {"acc": acc}


class SimilarityLoss(Loss):
    def forward(self, text_feats, img_feats, img_idxs, aux_input=None):
        sims = self.get_similarity_scores(text_feats, img_feats, img_idxs, aux_input)

        labels = torch.zeros(sims.shape[0]).long().to(img_feats.device)

        loss = -sims[:, 0]
        acc = (sims.argmax(dim=1) == labels).detach().float()

        return loss, {"acc": acc}


def get_loss(loss_type: str, dataset: str, num_hard_negatives: int):
    train_emb, train_nns = DATASET2NEG_PATHS.get(dataset.lower(), (None, None))

    name2loss = {
        "discriminative": DiscriminativeLoss,
        "accuracy": AccuracyLoss,
        "similarity": SimilarityLoss,
    }

    loss_cls = name2loss.get(loss_type.lower(), None)
    assert loss_cls, f"cannot recognize loss {loss_type}"

    return loss_cls(train_emb, train_nns, num_hard_negatives)
