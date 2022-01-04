def get_rf_opts(parser):
    pass
    # group = parser.add_argument_group("reinforce training options")
    # group.add_argument("--", type=float, default=None, help="")


def get_multi_symbol_opts(parser):
    group = parser.add_argument_group("multi symbol options")
    group.add_argument(
        "--sender_cell_dim",
        default=256,
        type=int,
        help="Size of sender hidden unit in recurrent network",
    )
    group.add_argument(
        "--recv_cell_dim",
        default=256,
        type=int,
        help="Size of recv hidden unit in recurrent network",
    )
    group.add_argument(
        "--sender_embed_dim",
        default=512,
        type=int,
        help="Size of sender embeddings in recurrent network",
    )
    group.add_argument(
        "--recv_embed_dim",
        default=512,
        type=int,
        help="Size of sender embeddings in recurrent network",
    )
    group.add_argument(
        "--sender_cell",
        default="rnn",
        choices=["rnn", "gru", "lstm"],
        help="Type of recurrent unit for generating a message in sender agent",
    )
    group.add_argument(
        "--recv_cell",
        default="rnn",
        choices=["rnn", "gru", "lstm"],
        help="Type of recurrent unit for generating a message in receiver agent",
    )
