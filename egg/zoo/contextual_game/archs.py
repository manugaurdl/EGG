# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from collections import Counter
from pathlib import Path
from typing import Callable

import clip
import numpy as np
import torch
import torch.nn as nn

from egg.core.gs_wrappers import RelaxedEmbedding, gumbel_softmax_sample
from egg.core.interaction import LoggingStrategy


class Sender(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(Sender, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
        )

    def forward(self, resnet_output, aux_input=None):
        return self.fc(resnet_output)


class ClipEmbeddingLoader:
    def __init__(
        self,
        pretrained_embeddings: torch.Tensor,
        freeze_embeddings: bool = False,
        max_vocab: int = None,
        data_path: str = "/private/home/rdessi/imagecode/data/",
    ):
        self.data_path = Path(data_path)

        assert max_vocab is None or max_vocab > 0

        self._load_embeddings(pretrained_embeddings, freeze_embeddings, max_vocab)

    def _load_embeddings(self, pretrained_embeddings, freeze_embeddings, max_vocab):
        # not including the test set since it is unlabeled and not used
        with open(self.data_path / "train_data.json") as fd:
            train = json.load(fd)
        with open(self.data_path / "valid_data.json") as fd:
            valid = json.load(fd)

        train_and_valid = {**train, **valid}

        token_list = []
        for _, captions in train_and_valid.items():
            for caption in captions.values():
                token_list.extend(clip.tokenize(caption, truncate=True)[0].tolist())

        token_counter = Counter(token_list)
        most_freq_tokens = [x[0] for x in token_counter.most_common(max_vocab)]

        self.embeddings = RelaxedEmbedding.from_pretrained(
            pretrained_embeddings.weight[most_freq_tokens],
            freeze=freeze_embeddings,
        )

        self.vocab_size = self.embeddings.weight.shape[0]


class RnnSenderFixedLengthGS(nn.Module):
    def __init__(
        self,
        agent: nn.Module,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        max_len: int,
        embeddings: nn.Module,
        cell: str = "rnn",
        temperature: float = 1.0,
        straight_through: bool = False,
    ):
        super(RnnSenderFixedLengthGS, self).__init__()
        self.agent = agent

        assert max_len >= 1, "Cannot have a max_len below 1"
        self.max_len = max_len

        self.hidden_to_output = nn.Linear(hidden_size, vocab_size)

        self.embedding = embeddings

        self.sos_embedding = nn.Parameter(torch.zeros(embed_dim))

        self.temperature = temperature
        self.straight_through = straight_through

        self.cell = None

        cell = cell.lower()

        if cell == "rnn":
            self.cell = nn.RNNCell(input_size=embed_dim, hidden_size=hidden_size)
        elif cell == "gru":
            self.cell = nn.GRUCell(input_size=embed_dim, hidden_size=hidden_size)
        elif cell == "lstm":
            self.cell = nn.LSTMCell(input_size=embed_dim, hidden_size=hidden_size)
        else:
            raise ValueError(f"Unknown RNN Cell: {cell}")

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.sos_embedding, 0.0, 0.01)

    def forward(self, x, aux_input=None):
        prev_hidden = self.agent(x, aux_input)
        prev_c = torch.zeros_like(prev_hidden)  # only for LSTM

        e_t = torch.stack([self.sos_embedding] * prev_hidden.size(0))
        sequence = []
        for step in range(self.max_len):
            if isinstance(self.cell, nn.LSTMCell):
                h_t, prev_c = self.cell(e_t, (prev_hidden, prev_c))
            else:
                h_t = self.cell(e_t, prev_hidden)

            step_logits = self.hidden_to_output(h_t)
            x = gumbel_softmax_sample(
                step_logits, self.temperature, self.training, self.straight_through
            )

            prev_hidden = h_t
            e_t = self.embedding(x)
            sequence.append(x)

        sequence = torch.stack(sequence).permute(1, 0, 2)

        return sequence


class ClipReceiver(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        embeddings: torch.Tensor,
        finetune_weights: bool,
        input_len: int = 77,  # clip defaul input len
    ):
        super(ClipReceiver, self).__init__()
        for name, param in model.named_parameters():
            if "visual" in name or "token_embedding" in name:
                continue
            param.requires_grad = True if finetune_weights else False
        model.token_embedding = embeddings
        self.model = model  # adding the model so its parameter are in the optimizer

        vocab_size = embeddings.weight.shape[0]
        # sos and eos idx are the last two in clip vocab
        self.sos_idx = vocab_size - 2
        self.eos_idx = vocab_size - 1

        self.input_len = input_len
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, message, image_features, aux_input=None):
        if len(message.shape) == 2:  # one-symbol messages
            message = message.unsqueeze(1)
        bsz, msg_len, embed_dim = message.shape

        out = torch.zeros(bsz, self.input_len, embed_dim, device=message.device)
        out[:, 1 : msg_len + 1] = message

        out[:, 0, self.sos_idx] = 1
        out[:, msg_len + 1, self.eos_idx] = 1

        text_features = self.model.encode_text(out)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = logit_scale * text_features @ image_features.t()
        return logits_per_text


class VisionGame(nn.Module):
    def __init__(
        self,
        sender_visual_encoder: nn.Module,
        recv_visual_encoder: nn.Module,
        sender: nn.Module,
        receiver: nn.Module,
        loss: Callable,
    ):
        super(VisionGame, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.loss = loss

        self.sender_visual_encoder = sender_visual_encoder
        self.recv_visual_encoder = recv_visual_encoder

        self.train_logging_strategy = LoggingStrategy().minimal()
        self.test_logging_strategy = LoggingStrategy()

    def forward(self, input_images, labels, receiver_input=None, aux_input=None):
        sender_input = self.sender_visual_encoder(input_images)
        recv_input = self.recv_visual_encoder(input_images)

        message = self.sender(sender_input, aux_input)
        receiver_output = self.receiver(message, recv_input, aux_input)

        loss, aux = self.loss(
            sender_input, message, receiver_input, receiver_output, labels, aux_input
        )

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )
        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input,
            receiver_input=receiver_input,
            labels=labels,
            aux_input=aux_input,
            receiver_output=receiver_output.detach(),
            message=message.detach(),
            message_length=torch.ones(message.size(0)),
            aux=aux,
        )
        return loss.mean(), interaction
