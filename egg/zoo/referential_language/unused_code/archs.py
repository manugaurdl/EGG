from typing import Optional

import torch.nn as nn

from egg.core.interaction import LoggingStrategy


class RnnReceiverFixedLengthGS(nn.Module):
    def __init__(self, agent, vocab_size, embed_dim, hidden_size, cell="rnn"):
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

        self.embedding = nn.Linear(vocab_size, embed_dim)

    def forward(self, message, input=None, aux_input=None):
        emb = self.embedding(message)

        prev_hidden = None
        prev_c = None

        # to get an access to the hidden states, we have to unroll the cell ourselves
        for step in range(message.size(1) - 1):
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

        return self.agent(h_t, input, aux_input)


class SenderReceiverRnnFixedLengthGS(nn.Module):
    def __init__(
        self,
        sender,
        receiver,
        loss,
        train_logging_strategy: Optional[LoggingStrategy] = None,
        test_logging_strategy: Optional[LoggingStrategy] = None,
    ):
        super(SenderReceiverRnnFixedLengthGS, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.loss = loss
        self.train_logging_strategy = (
            LoggingStrategy()
            if train_logging_strategy is None
            else train_logging_strategy
        )
        self.test_logging_strategy = (
            LoggingStrategy()
            if test_logging_strategy is None
            else test_logging_strategy
        )

    def forward(self, sender_input, labels, receiver_input=None, aux_input=None):
        message = self.sender(sender_input, aux_input)
        receiver_output = self.receiver(message, receiver_input, aux_input)

        loss, aux_info = self.loss(
            sender_input,
            message,
            receiver_input,
            receiver_output,
            labels,
            aux_input,
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
            message_length=None,
            aux=aux_info,
        )

        return loss.mean(), interaction
