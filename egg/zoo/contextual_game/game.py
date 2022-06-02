# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import clip
import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration, BartTokenizer

from typing import Callable

from egg.core.interaction import LoggingStrategy


class BartSender(nn.Module):
    def __init__(
        self,
        bart_model: str = "eugenesiow/bart-paraphrase",  # "facebook/bart-base"
        num_beams: int = 4,
    ):
        super(BartSender, self).__init__()
        self.bart_tokenizer = BartTokenizer.from_pretrained(bart_model)
        bart = BartForConditionalGeneration.from_pretrained(bart_model).eval()
        self.bart = bart

        self.num_beams = num_beams

    def forward(self, inputs, aux_input=None):
        # 75 is clip max_len w/o counting sos and eos
        generated_ids = self.bart.generate(
            inputs["input_ids"], num_beams=self.num_beams, max_length=75
        )

        generated_text = self.bart_tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return generated_text, generated_ids


class ClipReceiver(nn.Module):
    def __init__(
        self,
        clip_model: nn.Module,
    ):
        super(ClipReceiver, self).__init__()
        self.clip = clip_model
        self.clip.eval()

    def forward(self, message, images, aux_input=None):
        text = clip.tokenize(message, truncate=True).to(images.device)
        _, clip_logits = self.clip(images, text)
        return clip_logits


class Game(nn.Module):
    def __init__(
        self,
        sender: nn.Module,
        receiver: nn.Module,
        loss: Callable,
    ):
        super(Game, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.loss = loss

        self.train_logging_strategy = LoggingStrategy().minimal()
        self.test_logging_strategy = LoggingStrategy(
            True, False, True, True, True, True, False
        )

    def forward(self, sender_input, labels, receiver_input=None, aux_input=None):
        message, message_ids = self.sender(sender_input, aux_input)
        receiver_output = self.receiver(message, receiver_input, aux_input)

        loss, aux = self.loss(
            sender_input, message, receiver_input, receiver_output, labels, aux_input
        )

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )
        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input["input_ids"],
            receiver_input=receiver_input,
            labels=labels,
            aux_input=aux_input,
            receiver_output=receiver_output.detach(),
            message=message_ids.detach(),
            message_length=torch.ones(message_ids.size(0)),
            aux=aux,
        )
        return loss.sum(), interaction


def loss(
    _sender_input,
    _message,
    _receiver_input,
    receiver_output,
    labels,
    _aux_input,
):
    loss = torch.zeros(1).to(receiver_output.device)
    acc = (receiver_output.argmax(dim=1) == labels).detach().float()
    return loss, {"acc": acc}


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()


def build_game(opts):

    sender = BartSender(opts.bart_model, num_beams=opts.num_beams)

    clip_model = clip.load("ViT-B/16")[0]
    convert_models_to_fp32(clip_model)
    receiver = ClipReceiver(clip_model)

    game = Game(sender, receiver, loss)
    if opts.distributed_context.is_distributed:
        game = torch.nn.SyncBatchNorm.convert_sync_batchnorm(game)

    return game
