# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import json
from collections import Counter
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
        max_vocab: int = None,
        data_path: str = "/private/home/rdessi/imagecode/data/",
    ):
        self.data_path = Path(data_path)

        self.freeze_embeddings = freeze_embeddings

        self.clip_model = clip_model if clip_model else clip.load("ViT-B/16")[0]

        if max_vocab:
            assert max_vocab > 0
        self.max_vocab = max_vocab

        self._load_embeddings()

    def _load_embeddings(self):
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
        most_freq_tokens = [
            x[0]
            for x in token_counter.most_common(self.max_vocab + 3)
            # if x[0] not in [49406, 49407, 0]  # eos, sos and pad
        ]

        embeddings = self.clip_model.token_embedding
        self.embeddings = RelaxedEmbedding.from_pretrained(
            embeddings.weight[most_freq_tokens], freeze=self.freeze_embeddings
        )

        self.vocab_size, self.embedding_size = self.embeddings.weight.shape


class GumbelSoftmaxWrapper(nn.Module):
    """
    Gumbel-Softmax Wrapper for an agent that outputs a single symbol. Assumes that during the forward pass,
    the agent returns log-probabilities over the potential output symbols. During training, the wrapper
    transforms them into a sample from the Gumbel Softmax (GS) distribution;
    eval-time it returns greedy one-hot encoding of the same shape.
    """

    def __init__(
        self,
        agent,
        temperature=1.0,
        trainable_temperature=False,
        straight_through=False,
        **kwargs,
    ):
        super(GumbelSoftmaxWrapper, self).__init__()
        self.agent = agent
        self.straight_through = straight_through
        if not trainable_temperature:
            self.temperature = temperature
        else:
            self.temperature = torch.nn.Parameter(
                torch.tensor([temperature]), requires_grad=True
            )

    def forward(self, *args, **kwargs):
        logits = self.agent(*args, **kwargs)
        sample = gumbel_softmax_sample(
            logits, self.temperature, self.training, self.straight_through
        )
        return sample


class SymbolReceiverWrapper(nn.Module):
    """
    A wrapper for single-symbol Receiver, both Gumbel-Softmax and Reinforce. Receives a message, embeds it,
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
        agent: nn.Module,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        max_len: int,
        embeddings: Optional[nn.Module] = None,
        freeze_embeddings: bool = False,
        cell: str = "rnn",
        trainable_temperature: bool = False,
        temperature: float = 1.0,
        straight_through: bool = False,
    ):
        super(RnnSenderFixedLengthGS, self).__init__()
        self.agent = agent

        assert max_len >= 1, "Cannot have a max_len below 1"
        self.max_len = max_len

        self.hidden_to_output = nn.Linear(hidden_size, vocab_size)

        self.embedding = embeddings if embeddings else nn.Linear(vocab_size, embed_dim)

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
        embeddings: torch.Tensor = None,
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

        self.embedding = embeddings if embeddings else nn.Linear(vocab_size, embed_dim)

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


class ClipReceiver(nn.Module):
    def __init__(
        self,
        agent: nn.Module,
        model: Optional[nn.Module] = None,
        embeddings: Optional[torch.Tensor] = None,
        model_name: str = "ViT-B/16",
        add_clip_tokens: bool = False,
        input_len: int = 77,  # clip defaul input len
        finetune_weights: bool = False,
        freeze_embeddings: bool = False,
        max_clip_vocab: int = None,
    ):
        super(ClipReceiver, self).__init__()
        self.agent = agent
        if not model:
            model = clip.load(model_name)[0]

        for name, param in model.named_parameters():
            if "visual" in name or "token_embedding" in name:
                continue
            param.requires_grad = True if finetune_weights else False

        if embeddings is None:
            embeddings = ClipEmbeddingLoader(
                model, freeze_embeddings, max_clip_vocab
            ).embeddings

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

        return self.agent(self.text_encoder(out), image_features, aux_input)


class VisionGame(nn.Module):
    def __init__(
        self,
        visual_encoder: nn.Module,
        sender: nn.Module,
        receiver: nn.Module,
        loss: Callable,
        freeze_vision: Optional[str] = None,
    ):
        super(VisionGame, self).__init__()
        self._setup_visual_encoders(visual_encoder, freeze_vision)

        self.sender = sender
        self.receiver = receiver
        self.loss = loss

        self.train_logging_strategy = LoggingStrategy().minimal()
        self.test_logging_strategy = LoggingStrategy()

    def _setup_visual_encoders(self, visual_encoder, freeze_vision):
        if freeze_vision:
            if freeze_vision == "both":
                visual_encoder.eval()
                for param in visual_encoder.parameters():
                    param.requires_grad = False
                self.sender_visual_encoder = self.recv_visual_encoder = visual_encoder

            else:
                model_w_grad = copy.deepcopy(visual_encoder)
                model_w_grad.train()

                for param in visual_encoder.parameters():
                    param.requires_grad = False
                model_wo_grad = visual_encoder
                model_wo_grad.eval()

                if freeze_vision == "recv_only":
                    self.sender_visual_encoder = model_w_grad
                    self.recv_visual_encoder = model_wo_grad
                else:  # freeze_vision == sender_only
                    self.sender_visual_encoder = model_wo_grad
                    self.recv_visual_encoder = model_w_grad

        else:  # train_both (shared vision)
            visual_encoder.train()
            self.sender_visual_encoder = self.recv_visual_encoder = visual_encoder

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
