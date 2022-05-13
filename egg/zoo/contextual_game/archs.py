# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from collections import Counter
from pathlib import Path
from typing import Callable

import clip
import torch
import torch.nn as nn

from egg.core.gs_wrappers import RelaxedEmbedding, gumbel_softmax_sample as gs
from egg.core.interaction import LoggingStrategy


class ClipEmbeddingLoader:
    def __init__(
        self,
        pretrained_embeddings: torch.Tensor,
        freeze_embeddings: bool = False,
        max_vocab: int = None,
        include_special_symbols: bool = False,
        data_path: str = "/private/home/rdessi/imagecode/data/",
    ):
        self.data_path = Path(data_path)

        assert max_vocab is None or max_vocab > 0

        self.include_special_symbols = include_special_symbols

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

        max_vocab = max_vocab if max_vocab else len(token_counter)
        most_freq_tokens = [
            x[0]
            for x in token_counter.most_common(max_vocab + 3)
            if x[0] not in [49406, 49407, 0]  # eos, sos and pad
        ]

        if self.include_special_symbols:
            most_freq_tokens.extend([0, 49406, 49407])

        self.embeddings = RelaxedEmbedding.from_pretrained(
            pretrained_embeddings.weight[most_freq_tokens],
            freeze=freeze_embeddings,
        )


class SymbolSender(nn.Module):
    def __init__(
        self,
        agent: nn.Module,
        fc: nn.Module,
        temperature=1.0,
        straight_through=False,
        **kwargs,
    ):
        super(SymbolSender, self).__init__()
        self.agent = agent
        self.fc = fc

        self.straight_through = straight_through
        self.temperature = temperature

    def forward(self, image_features, aux_input=None):
        x = self.agent(image_features)
        logits = self.fc(x)
        return gs(logits, self.temperature, self.training, self.straight_through)


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
        self.max_len = max_len

        self.hidden_to_output = nn.Linear(hidden_size, vocab_size)

        self.embedding = embeddings

        self.sos_embedding = nn.Parameter(torch.zeros(embed_dim))

        self.temperature = temperature
        self.straight_through = straight_through

        name2cell = {"rnn": nn.RNNCell, "gru": nn.GRUCell, "lstm": nn.LSTMCell}

        self.cell = name2cell[cell.lower()](
            input_size=embed_dim, hidden_size=hidden_size
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.sos_embedding, 0.0, 0.01)

    def forward(self, image_features, aux_input=None):
        prev_hidden = self.agent(image_features)
        prev_c = torch.zeros_like(prev_hidden)  # only for LSTM

        e_t = torch.stack([self.sos_embedding] * prev_hidden.size(0))
        sequence = []
        for step in range(self.max_len):
            if isinstance(self.cell, nn.LSTMCell):
                h_t, prev_c = self.cell(e_t, (prev_hidden, prev_c))
            else:
                h_t = self.cell(e_t, prev_hidden)

            step_logits = self.hidden_to_output(h_t)
            x = gs(step_logits, self.temperature, self.training, self.straight_through)

            prev_hidden = h_t
            e_t = self.embedding(x)
            sequence.append(x)

        sequence = torch.stack(sequence).permute(1, 0, 2)

        return sequence


class InformedRnnSenderFixedLengthGS(nn.Module):
    def __init__(
        self,
        agent: nn.Module,
        vocab_size: int,
        embed_dim: int,  # must be the same as output dim of the agent
        hidden_size: int,  # must be the same as clip embedding dim (512)
        max_len: int,
        embeddings: nn.Module,
        cell: str = "rnn",
        temperature: float = 1.0,
        straight_through: bool = False,
    ):
        super(InformedRnnSenderFixedLengthGS, self).__init__()
        self.agent = agent
        self.max_len = max_len

        self.hidden_to_output = embeddings  # shape is hidden_size X vocab_size

        self.embedding = RelaxedEmbedding(vocab_size, embed_dim)

        self.prev_hidden = nn.Parameter(torch.zeros(hidden_size))

        self.temperature = temperature
        self.straight_through = straight_through

        name2cell = {"rnn": nn.RNNCell, "gru": nn.GRUCell, "lstm": nn.LSTMCell}

        self.cell = name2cell[cell.lower()](
            input_size=embed_dim, hidden_size=hidden_size
        )

    def forward(self, image_features, aux_input=None):
        img_feats = self.agent(image_features)

        prev_hidden = torch.stack([self.prev_hidden] * image_features.size(0))
        prev_c = torch.zeros_like(prev_hidden)  # only for LSTM

        sos_embedding = torch.zeros_like(img_feats)
        e_t = sos_embedding + img_feats

        sequence = []
        for step in range(self.max_len):
            if isinstance(self.cell, nn.LSTMCell):
                h_t, prev_c = self.cell(e_t, (prev_hidden, prev_c))
            else:
                h_t = self.cell(e_t, prev_hidden)

            step_logits = self.hidden_to_output(h_t)
            x = gs(step_logits, self.temperature, self.training, self.straight_through)

            prev_hidden = h_t
            e_t = self.embedding(x) + img_feats
            sequence.append(x)

        sequence = torch.stack(sequence).permute(1, 0, 2)

        return sequence


class ClipReceiver(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        embeddings: torch.Tensor,
        finetune_weights: bool,
        pad_idx: int = 49405,  # clip pad
        sos_idx: int = 49406,  # clip sos
        eos_idx: int = 49407,  # clip eos
        input_len: int = 77,  # clip defaul input len
    ):
        super(ClipReceiver, self).__init__()
        for name, param in model.named_parameters():
            if "visual" in name or "token_embedding" in name:
                continue
            param.requires_grad = True if finetune_weights else False

        model.token_embedding = embeddings
        self.model = model  # adding the model so its parameter are in the optimizer

        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx

        self.input_len = input_len

    def forward(self, message, image_features, aux_input=None):
        if len(message.shape) == 2:  # one-symbol messages
            message = message.unsqueeze(1)
        bsz, msg_len, embed_dim = message.shape

        out = torch.zeros(bsz, self.input_len, embed_dim, device=message.device)
        out[:, 1 : msg_len + 1] = message
        out = torch.cat(
            [out, torch.zeros(bsz, self.input_len, 3, device=message.device)], dim=-1
        )

        out[:, 0, self.sos_idx] = 1
        out[:, msg_len + 1, self.eos_idx] = 1
        out[:, msg_len + 2 :, self.pad_idx] = 1

        text_features = self.model.encode_text(out)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logits_per_text = text_features @ image_features.t()
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
