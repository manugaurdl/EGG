# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict

import clip
import torch
from torch import nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from egg.core.interaction import LoggingStrategy
from egg.zoo.contextual_game.archs import (
    ClipReceiver,
    MLP,
    TransformerMapper,
    generate_beam,
    generate2,
)


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


class ClipCaptionModel(nn.Module):
    def __init__(
        self,
        mapping_type: str,
        constant_prefix_tokens: int,
        clip_prefix_tokens: int,
        clip_prefix_size: int,
        num_layers: int,
    ):
        super(ClipCaptionModel, self).__init__()

        self.clip_prefix_tokens = clip_prefix_tokens
        self.gpt = GPT2LMHeadModel.from_pretrained("gpt2")
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]

        if mapping_type.lower() == "transformer":
            self.clip_project = TransformerMapper(
                clip_prefix_size,
                self.gpt_embedding_size,
                constant_prefix_tokens,
                clip_prefix_tokens,
                num_layers,
            )

        elif mapping_type.lower() == "mlp":
            if clip_prefix_tokens > 10:  # not enough memory
                input_dim = clip_prefix_size
                output_dim = self.gpt_embedding_size * clip_prefix_tokens
                self.clip_project = nn.Linear(input_dim, output_dim)
            else:
                input_dim = clip_prefix_size
                hidden_dim = (self.gpt_embedding_size * clip_prefix_tokens) // 2
                output_dim = self.gpt_embedding_size * clip_prefix_tokens
                self.clip_project = MLP((input_dim, hidden_dim, output_dim))
        else:
            raise RuntimeError("Cannor recognize {mapping_type}")

    def forward(
        self, image_feats: torch.Tensor, aux_input: Dict[Any, torch.Tensor] = None
    ):
        caption = aux_input["captions"].tolist()
        tokens = torch.Tensor([token for token in caption[0] if token >= 0])
        tokens = tokens.long().to(image_feats.device).unsqueeze(0)

        embedding_text = self.gpt.transformer.wte(tokens)
        prefix = self.clip_project(image_feats)
        prefix = prefix.view(-1, self.clip_prefix_tokens, self.gpt_embedding_size)

        dummy_tokens = torch.ones(tokens.shape[0], self.clip_prefix_tokens).long()
        dummy_tokens = dummy_tokens.to(tokens.device) * -100
        labels = torch.cat((dummy_tokens, tokens), dim=1)

        embedding_cat = torch.cat((prefix, embedding_text), dim=1)
        return self.gpt(inputs_embeds=embedding_cat, labels=labels)


class ClipCaptionModelPrefix(ClipCaptionModel):
    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def named_parameters(self, prefix="", recurse: bool = True):
        return self.clip_project.named_parameters()

    def train(self, mode: bool = True):
        self.training = mode
        self.clip_project.train(mode)
        self.gpt.eval()
        return self


class ClipClapSenderTrain(nn.Module):
    def __init__(
        self,
        mapping_type: str,
        constant_prefix_tokens: int,
        clip_prefix_tokens: int,
        clip_prefix_size: int,
        num_layers: int,
        model_path: str = None,
        clip_model: str = "ViT-B/32",
        use_beam_search: bool = True,
        num_beams: int = 5,
        prefix_only: bool = False,
    ):
        super(ClipClapSenderTrain, self).__init__()

        self.clip_model = clip.load(clip_model)[0]
        convert_models_to_fp32(self.clip_model)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        self.clip_prefix_tokens = clip_prefix_tokens

        model = ClipCaptionModelPrefix if prefix_only else ClipCaptionModel

        self.clipclap_model = model(
            mapping_type=mapping_type,
            constant_prefix_tokens=constant_prefix_tokens,
            clip_prefix_tokens=clip_prefix_tokens,
            clip_prefix_size=clip_prefix_size,
            num_layers=num_layers,
        )
        if model_path is not None:
            self.clipclap_model.load_state_dict(torch.load(model_path))

        self.use_beam_search = use_beam_search
        self.num_beams = num_beams

    def parameters(self, recurse: bool = True):
        return self.clipclap_model.parameters()

    def named_parameters(self, prefix="", recurse: bool = True):
        return self.clipclap_model.named_parameters()

    def train(self, mode: bool = True):
        self.training = mode
        self.clipclap_model.train(mode)
        self.clip_model.eval()
        return self

    def forward(self, image, aux_input=None):
        with torch.no_grad():
            prefix = self.clip_model.encode_image(image)

        prefix_embed = self.clipclap_model(prefix, aux_input)
        return prefix_embed.logits


class ClipClapSenderInference(ClipClapSenderTrain):
    def forward(self, image, aux_input=None):
        with torch.no_grad():
            prefix = self.clip_model.encode_image(image)
            prefix_embed = self.clipclap_model.clip_project(prefix)

        prefix_embed = prefix_embed.reshape(1, self.clip_prefix_tokens, -1)

        if self.use_beam_search:
            return generate_beam(
                self.clipclap_model,
                self.tokenizer,
                embed=prefix_embed,
                beam_size=self.num_beams,
            )[0]
        else:
            return generate2(self.clipclap_model, self.tokenizer, embed=prefix_embed)


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

        self.train_logging_strategy = LoggingStrategy.minimal()
        self.test_logging_strategy = LoggingStrategy(
            False, False, True, True, True, True, False
        )

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def parameters(self, recurse: bool = True):
        return self.sender.parameters()

    def named_parameters(self, prefix="", recurse: bool = True):
        return self.sender.named_parameters()

    def train(self, mode: bool = True):
        self.training = mode
        self.sender.train(mode)
        self.receiver.eval()
        return self

    def forward(self, sender_input, labels, receiver_input=None, aux_input=None):
        message = self.sender(sender_input, aux_input)
        receiver_output = self.receiver(message, receiver_input, aux_input)

        loss, aux = self.loss(
            sender_input, message, receiver_input, receiver_output, labels, aux_input
        )

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )

        if isinstance(message, str):
            message = torch.Tensor(self.tokenizer([message])["input_ids"]).long()

        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input,
            receiver_input=receiver_input,
            labels=labels,
            aux_input=aux_input,
            receiver_output=receiver_output.detach(),
            message=message.detach(),
            message_length=None,
            aux=aux,
        )
        return loss.mean(), interaction


def build_game(opts):
    clip_model = clip.load(opts.clip_model)[0]
    convert_models_to_fp32(clip_model)

    sender = ClipClapSenderInference(
        model_path=opts.clipclap_model_path,
        mapping_type=opts.mapping_type,
        constant_prefix_tokens=opts.constant_prefix_tokens,
        clip_prefix_tokens=opts.clip_prefix_tokens,
        clip_prefix_size=clip_model.visual.output_dim,
        num_layers=opts.num_transformer_layers,
        clip_model=opts.clip_model,
        use_beam_search=opts.use_beam_search,
        num_beams=opts.num_beams,
    )

    receiver = ClipReceiver(clip_model)

    game = Game(sender, receiver, loss)

    return game
