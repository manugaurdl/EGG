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
