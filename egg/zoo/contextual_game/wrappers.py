# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from pathlib import Path
from typing import Callable, Optional

import clip
import torch
import torch.nn as nn

from egg.core.gs_wrappers import RelaxedEmbedding, gumbel_softmax_sample
from egg.core.interaction import LoggingStrategy


class ClipEmbeddingLoader:
    def __init__(
        self,
        clip_model: Optional[nn.Module] = None,
        freeze_embeddings: bool = True,
        data_path: str = "/private/home/rdessi/imagecode/data/",
    ):
        self.data_path = Path(data_path)

        self.freeze_embeddings = freeze_embeddings
        self.tokenizer = clip.simple_tokenizer.SimpleTokenizer()

        self.clip_model = clip_model if clip_model else clip.load("ViT-B/16")[0]

        self._load_embeddings()

    def _load_embeddings(self):
        token_set = {
            self.tokenizer.encoder["<|startoftext|>"],
            self.tokenizer.encoder["<|endoftext|>"],
        }

        # not including the test set since it is unlabeled and not used
        with open(self.data_path / "train_data.json") as fd:
            train = json.load(fd)
        with open(self.data_path / "valid_data.json") as fd:
            valid = json.load(fd)

        train_and_valid = {**train, **valid}

        all_sents = []
        for _, captions in train_and_valid.items():
            for caption in captions.values():
                all_sents.append(caption)
                tokens = clip.tokenize(caption, truncate=True).unique().tolist()
                token_set.update(tokens)

        embeddings = self.clip_model.token_embedding

        token_list = list(token_set)
        token_list.sort()

        self.embeddings = RelaxedEmbedding.from_pretrained(
            embeddings.weight[token_list], freeze=self.freeze_embeddings
        )

        self.vocab_size, self.embedding_size = self.embeddings.weight.shape


class SymbolReceiverWrapper(nn.Module):
    """
    An optional wrapper for single-symbol Receiver, both Gumbel-Softmax and Reinforce. Receives a message, embeds it,
    and passes to the wrapped agent. Same as the one in core with the additional
    option of using pretrained embeddings
    """

    def __init__(
        self,
        agent: nn.Module,
        vocab_size: int,
        agent_input_size: int,
        embeddings: nn.Module = None,
    ):
        super(SymbolReceiverWrapper, self).__init__()
        self.agent = agent

        self.embedding = (
            embeddings if embeddings else RelaxedEmbedding(vocab_size, agent_input_size)
        )

    def forward(self, message, input=None, aux_input=None):
        embedded_message = self.embedding(message)
        return self.agent(embedded_message, input, aux_input)


class RnnSenderFixedLengthGS(nn.Module):
    def __init__(
        self,
        agent,
        vocab_size,
        embed_dim,
        hidden_size,
        max_len,
        temperature,
        pretrained_embeddings=None,
        freeze_embeddings=False,
        cell="rnn",
        trainable_temperature=False,
        straight_through=False,
    ):
        super(RnnSenderFixedLengthGS, self).__init__()
        self.agent = agent

        assert max_len >= 1, "Cannot have a max_len below 1"
        self.max_len = max_len

        self.hidden_to_output = nn.Linear(hidden_size, vocab_size)

        if pretrained_embeddings:
            self.embedding = pretrained_embeddings
        else:
            self.embedding = nn.Linear(vocab_size, embed_dim)

        self.sos_embedding = nn.Parameter(torch.zeros(embed_dim))

        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        if not trainable_temperature:
            self.temperature = temperature
        else:
            self.temperature = torch.nn.Parameter(
                torch.tensor([temperature]), requires_grad=True
            )

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
        model: Optional[nn.Module] = None,
        embeddings: Optional[torch.Tensor] = None,
        model_name: str = "VIT-B/16",
        add_clip_tokens: bool = False,
        input_len: int = 77,  # clip defaul input len
        finetune_weights: bool = False,
        freeze_embeddings: bool = False,
    ):
        super(ClipReceiver, self).__init__()
        if not model:
            model = clip.load(model_name)[0]

        for p in model.parameters():
            p.requires_grad = True if finetune_weights else False

        if embeddings is None:
            embeddings = ClipEmbeddingLoader(model, freeze_embeddings).embeddings

        model.token_embedding = embeddings
        self.model = model  # adding the model so its parameter are in the optimizer
        self.text_encoder = self.model.encode_text

        self.add_clip_tokens = add_clip_tokens
        if self.add_clip_tokens:
            vocab_size = embeddings.weight.shape[0]
            # sos and eos idx are the last two in clip vocab
            self.sos_idx = vocab_size - 2
            self.eos_idx = vocab_size - 1

        self.input_len = input_len

    def forward(self, message, image_features, aux_input=None):
        if len(message.shape) == 2:
            message = message.unsqueeze(1)
        bsz, msg_len, embed_dim = message.shape
        assert msg_len < self.input_len - 2  # counting sos and eos

        out = torch.zeros(bsz, self.input_len, embed_dim, device=message.device)
        out[:, 1 : msg_len + 1] = message

        if self.add_clip_tokens:
            out[:, 0, self.sos_idx] = 1
            out[:, msg_len + 1, self.eos_idx] = 1

        return self.text_encoder(out)


class RnnReceiverFixedLengthGS(nn.Module):
    """
    Gumbel Softmax-based wrapper for Receiver agent in fixed-length communication game. The user implemented logic
    is passed in `agent` and is responsible for mapping (RNN's hidden state + Receiver's optional input)
    into the output vector.
    """

    def __init__(
        self,
        agent: nn.Module,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        cell: str = "rnn",
        pretrained_embeddings: torch.Tensor = None,
        freeze_embeddings: bool = False,
    ):
        super(RnnReceiverFixedLengthGS, self).__init__()
        self.agent = agent

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

        if pretrained_embeddings:
            self.embedding = pretrained_embeddings
        else:
            self.embedding = nn.Linear(vocab_size, embed_dim)

    def forward(self, message, input=None, aux_input=None):
        emb = self.embedding(message)

        prev_hidden = None
        prev_c = None

        # to get an access to the hidden states, we have to unroll the cell ourselves
        for step in range(message.size(1)):
            e_t = emb[:, step, ...]
            if isinstance(self.cell, nn.LSTMCell):
                h_t, prev_c = (
                    self.cell(e_t, (prev_hidden, prev_c))
                    if prev_hidden is not None
                    else self.cell(e_t)
                )
            else:
                h_t = self.cell(e_t, prev_hidden)

            prev_hidden = h_t

        output = self.agent(h_t, input, aux_input)

        return output


class VisionGame(nn.Module):
    def __init__(
        self,
        visual_encoder: nn.Module,
        sender: nn.Module,
        receiver: nn.Module,
        loss: Callable,
    ):
        super(VisionGame, self).__init__()
        self.visual_encoder = visual_encoder

        self.sender = sender
        self.receiver = receiver
        self.loss = loss

        self.train_logging_strategy = LoggingStrategy().minimal()
        self.test_logging_strategy = LoggingStrategy()

    def forward(self, sender_input, labels, receiver_input=None, aux_input=None):
        # TODO: currently there's no support for non-shared vision modules
        visual_feats = self.visual_encoder(sender_input)

        message = self.sender(visual_feats, aux_input)
        receiver_output = self.receiver(message, visual_feats, aux_input)

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
