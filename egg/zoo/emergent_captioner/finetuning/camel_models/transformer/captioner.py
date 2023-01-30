import torch
from torch import Tensor, nn

from egg.zoo.emergent_captioner.finetuning.camel_models.beam_search import *
from egg.zoo.emergent_captioner.finetuning.camel_models.containers import Module
from egg.zoo.emergent_captioner.finetuning.camel_models.transformer import (
    Decoder,
    Encoder,
    MeshedDecoder,
    ScaledDotProductAttentionMemory,
)
from egg.zoo.emergent_captioner.finetuning.camel_models.utils import TensorOrSequence


class Captioner(Module):
    def __init__(self, args, tokenizer):
        super(Captioner, self).__init__()

        self.encoder = Encoder(
            args.N_enc,
            500,
            args.image_dim,
            d_model=args.d_model,
            d_ff=args.d_ff,
            h=args.head,
            attention_module=ScaledDotProductAttentionMemory,
            attention_module_kwargs={"m": args.m},
            with_pe=args.with_pe,
            with_mesh=not args.disable_mesh,
        )
        if args.disable_mesh:
            self.decoder = Decoder(
                tokenizer.vocab_size,
                40,
                args.N_dec,
                d_model=args.d_model,
                d_ff=args.d_ff,
                h=args.head,
            )
        else:
            self.decoder = MeshedDecoder(
                tokenizer.vocab_size,
                40,
                args.N_dec,
                args.N_enc,
                d_model=args.d_model,
                d_ff=args.d_ff,
                h=args.head,
            )
        self.bos_idx = tokenizer.bos_idx
        self.eos_idx = tokenizer.eos_idx
        self.vocab_size = tokenizer.vocab_size
        self.max_generation_length = self.decoder.max_len

        self.register_state("enc_output", None)
        self.register_state("mask_enc", None)
        self.init_weights()

    @property
    def d_model(self):
        return self.decoder.d_model

    def train(self, mode: bool = True):
        self.encoder.train(mode)
        self.decoder.train(mode)

    def init_weights(self):
        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, images, seq):
        enc_output, mask_enc = self.encoder(images)
        dec_output = self.decoder(seq, enc_output, mask_enc)
        return dec_output

    def step(self, t: int, prev_output: Tensor, visual: Tensor) -> Tensor:
        if t == 0:
            self.enc_output, self.mask_enc = self.encoder(visual)
            input = visual.data.new_full(
                (visual.shape[0], 1), self.bos_idx, dtype=torch.long
            )
        else:
            input = prev_output
        logits = self.decoder(input, self.enc_output, self.mask_enc)
        return logits

    def beam_search(
        self,
        visual: TensorOrSequence,
        beam_size: int,
        out_size=1,
        return_logits=False,
        **kwargs
    ):
        bs = BeamSearch(self, self.max_generation_length, self.eos_idx, beam_size)
        return bs.apply(visual, out_size, return_logits, **kwargs)
