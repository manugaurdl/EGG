from egg.core.gs_wrappers import RnnSenderGS
from egg.zoo.referential_language.archs import (
    Receiver,
    RnnReceiverFixedLengthGS,
    Sender,
    SenderReceiverRnnFixedLengthGS,
)
from egg.zoo.referential_language.games import (
    get_logging_strategies,
    get_vision_modules,
    loss,
)


def build_gs_game(opts):
    train_logging_strategy, test_logging_strategy = get_logging_strategies()
    vision_module_sender, vision_module_receiver, sender_input_dim = get_vision_modules(
        opts
    )
    if opts.max_len > 1:
        sender = Sender(
            vision_module=vision_module_sender,
            input_dim=sender_input_dim,
            output_dim=opts.sender_cell_dim,
            num_heads=opts.num_heads,
            attention_type=opts.attention_type,
            context_integration=opts.context_integration,
            residual=opts.residual,
        )
        receiver = Receiver(
            vision_module=vision_module_receiver,
            input_dim=sender_input_dim,
            hidden_dim=opts.recv_hidden_dim,
            output_dim=opts.recv_cell_dim,
            temperature=opts.recv_temperature,
            use_cosine_sim=opts.cosine_similarity,
        )
        sender = RnnSenderGS(
            agent=sender,
            vocab_size=opts.vocab_size,
            embed_dim=opts.sender_embed_dim,
            hidden_size=opts.sender_cell_dim,
            max_len=opts.max_len - 1,
            temperature=opts.gs_temperature,
            cell=opts.sender_cell,
        )
        receiver = RnnReceiverFixedLengthGS(
            agent=receiver,
            vocab_size=opts.vocab_size,
            embed_dim=opts.recv_embed_dim,
            hidden_size=opts.recv_cell_dim,
            cell=opts.recv_cell,
        )
        game = SenderReceiverRnnFixedLengthGS(
            sender=sender,
            receiver=receiver,
            loss=loss,
            train_logging_strategy=train_logging_strategy,
            test_logging_strategy=test_logging_strategy,
        )
        return game


def build_rf_game(opts):
    raise NotImplementedError
