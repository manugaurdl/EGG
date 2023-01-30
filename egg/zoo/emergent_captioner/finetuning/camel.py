# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from itertools import takewhile
from typing import Any, Dict

import torch
import torch.nn as nn

from egg.zoo.emergent_captioner.finetuning.camel_models.camel_clip import (
    load as load_clip,
)
from egg.zoo.emergent_captioner.finetuning.camel_models.tokenizer.simple_tokenizer import (
    SimpleTokenizer as tokenizer,
)
from egg.zoo.emergent_captioner.finetuning.camel_models.transformer.captioner import (
    Captioner,
)
from egg.zoo.emergent_captioner.utils import convert_models_to_fp32


class CamelSender(nn.Module):
    def __init__(
        self,
        args,
        clip_model: str,
        camel_path: str,
        beam_size: int = 5,
        max_len: int = 20,
    ):
        super(CamelSender, self).__init__()
        assert max_len < 75  # clip maximum context size

        self.clip_vit = load_clip(clip_model, jit=False)[0].visual
        convert_models_to_fp32(self.clip_vit)
        self.clip_vit.eval()

        for p in self.clip_vit.parameters():
            p.requires_grad = False

        args.image_dim = self.clip_vit.embed_dim
        self.tokenizer = tokenizer()
        self.camel = Captioner(args, tokenizer=self.tokenizer)

        assert camel_path

        data = torch.load(camel_path)
        # if args.network == "target":
        #    self.camel.load_state_dict(data["state_dict_t"])
        # else:  # args.network == 'online'
        self.camel.load_state_dict(data["state_dict_o"])

        self.camel.train()

        print("| LOADED CAMEL MODEL")

        self.max_len = max_len
        self.beam_size = beam_size

    def _decode(self, word_idxs):
        if isinstance(word_idxs, list) and len(word_idxs) == 0:
            return self._decode(
                [
                    word_idxs,
                ]
            )[0]
        if isinstance(word_idxs, list) and isinstance(word_idxs[0], int):
            return self._decode(
                [
                    word_idxs,
                ]
            )[0]
        elif isinstance(word_idxs, torch.Tensor) and word_idxs.ndimension() == 1:
            return self._decode(word_idxs.unsqueeze(0))[0]

        captions = []
        for wis in word_idxs:
            wis = wis.tolist()
            wis = list(takewhile(lambda tok: tok != self.tokenizer.eos_idx, wis))
            caption = self.tokenizer.decode(wis)
            captions.append(caption)
        return captions

    def forward(self, images: torch.Tensor, aux_input: Dict[Any, torch.Tensor] = None):
        image_feats = self.clip_vit.intermediate_features(images)

        captions_idx, log_probs = self.camel.beam_search(
            image_feats, beam_size=self.beam_size
        )

        log_probs = torch.sum(log_probs, -1) / torch.sum(log_probs != 0, -1)

        captions = self._decode(captions_idx)

        return captions, log_probs, torch.tensor([0.0]).to(image_feats.device)
