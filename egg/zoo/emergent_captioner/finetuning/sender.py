# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from itertools import chain
from typing import Any, Dict, Tuple

import clip
import torch
import torch.nn as nn
from torch.distributions import Categorical
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    LogitsProcessor,
    LogitsProcessorList,
)

from egg.zoo.emergent_captioner.utils import convert_models_to_fp32


class KLRegularizer:
    def __init__(self, device=torch.device("cuda")):
        self.lm = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

    @torch.no_grad
    def compute_kl_loss(self, indices, log_probs):
        # 50256 is gpt2 beginning of sentence
        indices = torch.cat([torch.ones_like(indices[:, :1]) * 50256, indices], dim=1)
        # we take probs from bos until last token
        generated = self.lm(indices)["logits"].log_softmax(-1)[:, :-1, :]

        step_kl_div = []
        for timestep in range(generated.shape[1]):
            x = torch.nn.functional.kl_div(
                log_probs[:, timestep],
                generated[:, timestep],
                log_target=True,
                reduction="none",
            )
            step_kl_div.append(x.sum(-1))  # summing over vocab_dim
        kl_div = torch.stack(step_kl_div, dim=1)
        return kl_div


class StopTokenLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, do_sample):
        self.eos_token_id = tokenizer.eos_token_id

        self.stop_word_ids = set(
            [
                idx
                for idx in range(len(tokenizer))
                if "." in tokenizer.convert_ids_to_tokens(idx)
            ]
        )
        self.vocab_size = len(tokenizer)

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        for i, input_id in enumerate(input_ids):
            if input_id[-1].item() in self.stop_word_ids:
                scores[i, : self.vocab_size] = torch.finfo().min
                scores[i, self.vocab_size :] = float("-inf")
                scores[i, self.eos_token_id] = 0.0
        return scores


class MLP(nn.Module):
    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ClipCapModel(nn.Module):
    def __init__(
        self,
        clip_prefix_size: int,
        freeze_mapper: bool,
        num_return_sequences: int = 1,
        nb_prefix_tokens: int = 10,
        do_sample: bool = False,
        beam_size: int = 5,
        max_len: int = 20,
    ):
        super(ClipCapModel, self).__init__()

        self.gpt = GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.gpt.config.pad_token_id = self.gpt.config.eos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

        self.num_return_sequences = num_return_sequences
        self.do_sample = do_sample
        self.beam_size = beam_size
        self.max_len = max_len

        self.logits_processor = StopTokenLogitsProcessor(self.tokenizer, do_sample)

        gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        input_dim = clip_prefix_size

        self.nb_prefix_tokens = nb_prefix_tokens

        hidden_dim = (gpt_embedding_size * self.nb_prefix_tokens) // 2
        output_dim = gpt_embedding_size * self.nb_prefix_tokens
        self.clip_project = MLP((input_dim, hidden_dim, output_dim))

        self.freeze_mapper = freeze_mapper

        self.kl_regularizer = KLRegularizer()

    def forward(self, image_feats, aux_input=None):
        prompts = self.clip_project(image_feats)
        prompts = prompts.view(image_feats.shape[0], self.nb_prefix_tokens, -1)

        bsz, prefix_len, h_dim = prompts.shape

        prompts_flat = prompts.view(-1, prompts.size(-1))
        start = len(self.tokenizer)
        end = start + (bsz * prefix_len)
        self.gpt.get_input_embeddings().weight.data[start:end] = prompts_flat
        input_ids = torch.arange(start, end).view(*prompts.shape[:2]).to(prompts.device)

        if self.training:
            generated = self.gpt.generate(
                input_ids,
                do_sample=self.do_sample,
                max_length=self.max_len,
                num_beams=self.beam_size,
                num_return_sequences=self.num_return_sequences,
                logits_processor=LogitsProcessorList([self.logits_processor]),
                top_k=len(self.tokenizer),
            )
        else:
            # at test time we use beam search regardless of the decoding method
            # used at training time
            generated = self.gpt.generate(
                input_ids,
                do_sample=False,
                max_length=self.max_len,
                num_beams=5,
                num_return_sequences=1,
                logits_processor=LogitsProcessorList([self.logits_processor]),
                top_k=len(self.tokenizer),
            )

        if self.training:
            prompts = prompts.repeat_interleave(self.num_return_sequences, 0)

        indices = generated[:, prefix_len:]
        suffix = self.gpt.get_input_embeddings()(indices)
        inputs_embeds = torch.cat([prompts, suffix], dim=1)
        logits = self.gpt(inputs_embeds=inputs_embeds)
        logits = logits[0][:, prefix_len - 1 : -1, : len(self.tokenizer)]
        logits = logits.log_softmax(-1)

        # compute_mask and msg_lengths
        max_k = indices.size(1)
        end_of_caption = indices == self.eos_token_id
        extra_tokens = end_of_caption.cumsum(dim=1) > 0
        msg_lengths = max_k - (extra_tokens).sum(dim=1)
        msg_lengths.add_(1).clamp_(max=max_k)

        mask = (extra_tokens == 0).float()

        # compute normalized log_probs of generated captions
        log_probs = torch.gather(logits, dim=2, index=indices.unsqueeze(2)).squeeze()
        log_probs *= mask
        log_probs = log_probs.sum(1) / msg_lengths

        # compute avg per timestep entropy of captioner distribution
        entropies = []
        for timestep in range(logits.shape[1]):
            dist = Categorical(probs=logits[:, timestep].detach())
            ent = dist.entropy() * mask[:, timestep]
            entropies.append(ent)

        entropy = torch.stack(entropies, dim=1)
        entropy = entropy.sum(-1) / msg_lengths

        # put captions in textual form
        decoded_captions = self.tokenizer.batch_decode(
            indices,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        # compute kl loss
        kl_div = self.kl_regularizer.compute_kl_loss(indices, logits)
        kl_div *= mask
        kl_div = kl_div.sum(-1) / msg_lengths

        return decoded_captions, log_probs, entropy, kl_div

    def maybe_patch_gpt(self, max_embeddings):
        if not getattr(self.gpt, "_patched", False):
            self.gpt._patched = True
            self.gpt.resize_token_embeddings(len(self.tokenizer) + max_embeddings)
            if self.gpt.get_output_embeddings().bias is None:
                self.gpt.get_output_embeddings().bias = torch.nn.Parameter(
                    torch.tensor([0.0] * (len(self.tokenizer) + max_embeddings))
                )
                self.gpt.get_output_embeddings().bias.requires_grad = False
                self.gpt.get_output_embeddings().to(
                    self.gpt.get_output_embeddings().weight.device
                )
                self.gpt._originally_with_no_bias = True
            else:
                self.gpt._originally_with_no_bias = False
            self.gpt.get_output_embeddings().bias.data[-max_embeddings:] = float("-inf")

    def maybe_unpatch_gpt(self):
        if getattr(self.gpt, "_patched", False):
            self.gpt._patched = False
            self.gpt.resize_token_embeddings(len(self.tokenizer))
            if self.gpt._originally_with_no_bias:
                self.gpt.get_output_embeddings().bias = None

    def named_parameters(self, prefix="", recurse: bool = True):
        if self.freeze_mapper:
            return self.gpt.named_parameters()
        return chain(self.gpt.named_parameters(), self.clip_project.named_parameters())

    def parameters(self, recurse: bool = True):
        if self.freeze_mapper:
            return self.gpt.parameters()
        return chain(self.gpt.parameters(), self.clip_project.parameters())

    def train(self, mode: bool = True):
        self.training = mode
        if self.freeze_mapper:
            self.clip_project.eval()
        else:
            self.clip_project.train(mode)
        self.gpt.train(mode)
        return self


class ClipCapSender(nn.Module):
    def __init__(
        self,
        clip_model: str,
        clipcap_path: str,
        freeze_clipcap_mapper: bool = False,
        num_return_sequences: int = 1,
        do_sample: bool = False,
        beam_size: int = 5,
        max_len: int = 20,
    ):
        super(ClipCapSender, self).__init__()

        assert max_len < 75  # clip maximum context size

        self.clip_vit = clip.load(clip_model)[0].visual
        convert_models_to_fp32(self.clip_vit)
        self.clip_vit.eval()

        for p in self.clip_vit.parameters():
            p.requires_grad = False

        self.clipcap = ClipCapModel(
            clip_prefix_size=self.clip_vit.output_dim,
            freeze_mapper=freeze_clipcap_mapper,
            num_return_sequences=num_return_sequences,
            do_sample=do_sample,
            beam_size=beam_size,
            max_len=max_len,
        )
        if clipcap_path is not None:
            print("| LOADED CLIPCAP MODEL")
            self.clipcap.load_state_dict(torch.load(clipcap_path))

        self.patch_model()

    def encode_images(self, images: torch.Tensor):
        return self.clip_vit(images)

    def named_parameters(self, prefix="", recurse: bool = True):
        return self.clipcap.named_parameters()

    def parameters(self, recurse: bool = True):
        return self.clipcap.parameters()

    def train(self, mode: bool = True):
        self.training = mode
        self.clip_vit.eval()
        self.clipcap.train(mode)
        return self

    def patch_model(self, batch_size: int = 500, nb_prefix_tokens: int = 10):
        self.clipcap.maybe_patch_gpt(batch_size * nb_prefix_tokens)

    def unpatch_model(self):
        self.clipcap.maybe_unpatch_gpt()

    def forward(self, images: torch.Tensor, aux_input: Dict[Any, torch.Tensor] = None):
        image_feats = self.encode_images(images)
        captions, log_probs, entropy, kl_div = self.clipcap(image_feats, aux_input)
        return captions, log_probs, entropy, kl_div
