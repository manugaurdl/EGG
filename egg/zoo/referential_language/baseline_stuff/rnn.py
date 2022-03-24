import random

import torch
import torch.nn as nn


class MessageGeneratorRNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        temperature: int,
        nos: int,
        shuffle_cat: bool = False,
        cell: str = "rnn",
    ):
        super(MessageGeneratorRNN, self).__init__()

        self.hidden_to_output = nn.Linear(hidden_size, vocab_size)
        self.sos_embedding = nn.Parameter(torch.zeros(embed_dim))
        self.embedding = nn.Linear(hidden_size, embed_dim)

        self.shuffle_cat = shuffle_cat

        self.temperature = temperature

        name2cell = {"rnn": nn.RNNCell, "gru": nn.GRUCell}
        cell = cell.lower()
        self.cell = name2cell[cell](input_size=embed_dim, hidden_size=hidden_size)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.sos_embedding, 0.0, 0.01)

    def forward(self, target, ctx=None, aux_input=None):
        bsz, max_objs, _ = target.shape

        ctx = torch.Tensor([]) if ctx is None else ctx.contiguouos()

        inp_list = [target, ctx]
        random.shuffle(inp_list) if self.shuffle_cat else None

        sequence = []
        prev_hidden = torch.cat(inp_list, dim=-1).view(bsz * max_objs, -1)
        e_t = torch.stack([self.sos_embedding] * prev_hidden.size(0))
        for step in range(self.nos):
            h_t = self.cell(e_t, prev_hidden)
            step_logits = self.hidden_to_output(h_t)
            x = gumbel_softmax_sample(step_logits, self.temperature, self.training)

            prev_hidden = h_t
            e_t = self.embedding(x)
            sequence.append(x)

        return torch.stack(sequence).permute(1, 0, 2)
