# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable

import torch
import torch.nn as nn

from egg.core.gs_wrappers import RelaxedEmbedding, gumbel_softmax_sample as gs
from egg.core.interaction import LoggingStrategy


class InformedRnnSenderFixedLengthGS(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_encoder_layers: int,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        max_len: int,
        embeddings: nn.Module,
        cell: str = "rnn",
        temperature: float = 1.0,
        straight_through: bool = False,
    ):
        super(InformedRnnSenderFixedLengthGS, self).__init__()
        self.max_len = max_len

        # shape is hidden_size X vocab_size
        self.hidden_to_output = embeddings.weight.t()

        self.embedding = RelaxedEmbedding(vocab_size, embed_dim)

        self.prev_hidden = nn.Parameter(torch.zeros(hidden_size))

        self.temperature = temperature
        self.straight_through = straight_through

        name2cell = {"rnn": nn.RNNCell, "gru": nn.GRUCell}

        self.cell = name2cell[cell.lower()](
            input_size=embed_dim, hidden_size=hidden_size
        )

        encoder_hidden_sizes = [input_dim] * num_encoder_layers
        encoder_layer_dimensions = [(input_dim, encoder_hidden_sizes[0])]

        for i, hidden_size in enumerate(encoder_hidden_sizes[1:]):
            hidden_shape = (encoder_hidden_sizes[i], hidden_size)
            encoder_layer_dimensions.append(hidden_shape)

        encoder_layer_dimensions.append((input_dim, embed_dim))

        self.encoder_hidden_layers = nn.ModuleList(
            [nn.Linear(*dimensions) for dimensions in encoder_layer_dimensions]
        )

    def forward(self, x, aux_input=None):
        for hidden_layer in self.encoder_hidden_layers[:-1]:
            x = torch.tanh(hidden_layer(x))
        img_feats = self.encoder_hidden_layers[-1](x)

        prev_hidden = torch.stack([self.prev_hidden] * x.size(0))

        sos_embedding = torch.zeros_like(img_feats)
        e_t = sos_embedding * img_feats

        sequence = []
        for step in range(self.max_len):
            h_t = self.cell(e_t, prev_hidden)

            step_logits = h_t @ self.hidden_to_output

            x = gs(step_logits, self.temperature, self.training, self.straight_through)

            prev_hidden = h_t
            e_t = self.embedding(x) * img_feats

            sequence.append(x)

        sequence = torch.stack(sequence).permute(1, 0, 2)

        return sequence


class ClipReceiver(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        pad_idx: int,
        sos_idx: int,
        eos_idx: int,
        input_len: int = 77,  # clip defaul input len
    ):
        super(ClipReceiver, self).__init__()
        self.model = model

        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx

        self.input_len = input_len

    def encode_images(self, images):
        return self.model.visual(images)

    def forward(self, message, image_features, aux_input=None):
        bsz, msg_len, embed_dim = message.shape

        out = torch.zeros(bsz, self.input_len, embed_dim, device=message.device)
        out[:, 1 : msg_len + 1] = message

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
        sender: nn.Module,
        receiver: nn.Module,
        loss: Callable,
    ):
        super(VisionGame, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.loss = loss

        self.train_logging_strategy = LoggingStrategy().minimal()
        self.test_logging_strategy = LoggingStrategy(
            False, False, True, True, True, True, False
        )

    def forward(self, input_images, labels, receiver_input=None, aux_input=None):
        image_feats = self.receiver.encode_images(input_images)

        message = self.sender(image_feats, aux_input)
        receiver_output = self.receiver(message, image_feats, aux_input)

        loss, aux = self.loss(
            image_feats, message, image_feats, receiver_output, labels, aux_input
        )

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )
        interaction = logging_strategy.filtered_interaction(
            sender_input=image_feats,
            receiver_input=image_feats,
            labels=labels,
            aux_input=aux_input,
            receiver_output=receiver_output.detach(),
            message=message.detach(),
            message_length=torch.ones(message.size(0)),
            aux=aux,
        )
        return loss.sum(), interaction
