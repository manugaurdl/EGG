from typing import Any, Dict

import torch
import torch.nn as nn


class BLIPSender(nn.Module):
    def __init__(
        self,
        blip_model: str = "base_coco",
        freeze_image_encoder: bool = False,
        num_return_sequences: int = 1,
        do_sample: bool = False,
        beam_size: int = 5,
        max_len: int = 20,
    ):
        super(BLIPSender, self).__init__()

        assert not do_sample
        from lavis.models import load_model_and_preprocess

        self.do_sample = do_sample
        self.beam_size = beam_size
        self.num_return_sequences = num_return_sequences

        model, vis_processors, _ = load_model_and_preprocess(
            name="blip_caption", model_type=blip_model
        )
        print("| LOADED BLIP MODEL")

        self.transform = vis_processors["eval"]
        self.tokenizer = model.tokenizer
        self.prompt = self.tokenizer(model.prompt).input_ids[:-1]
        self.prompt[0] = self.tokenizer.bos_token_id

        self.model_txt = model.text_decoder
        self.model_txt.config.eos_token_id = self.tokenizer.sep_token_id
        self.model_txt.config.pad_token_id = self.tokenizer.pad_token_id
        self.model_txt.config.max_length = max_len

        self.model_img = model.visual_encoder
        if freeze_image_encoder:
            for p in self.model_img.parameters():
                p.requires_grad = False

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def prefix_len(self):
        return len(self.prompt)

    def forward(self, images: torch.Tensor, aux_input: Dict[Any, torch.Tensor] = None):

        with torch.no_grad():
            feats_img = self.model_img(images)
            feats_img = feats_img.repeat_interleave(self.beam_size, 0)
            attns_img = torch.ones(feats_img.size()[:-1], dtype=torch.long).to(
                self.device
            )

            model_kwargs = {
                "encoder_hidden_states": feats_img,
                "encoder_attention_mask": attns_img,
            }

            prompts = (
                torch.tensor(self.prompt)
                .unsqueeze(0)
                .repeat(images.size(0), 1)
                .to(self.device)
            )

            captions = self.model_txt.generate(
                input_ids=prompts,
                num_beams=self.beam_size,
                do_sample=self.do_sample,
                num_return_sequences=self.num_return_sequences,
                early_stopping=False,
                **model_kwargs
            )

            mask = captions != self.model_txt.config.pad_token_id
            mask = mask[:, self.prefix_len :]
            msg_lengths = mask.long().sum(-1)

        feats_img = self.model_img(images)
        feats_img = feats_img.repeat_interleave(self.num_return_sequences, 0)
        attns_img = torch.ones(feats_img.size()[:-1], dtype=torch.long).to(self.device)

        model_kwargs = {
            "encoder_hidden_states": feats_img,
            "encoder_attention_mask": attns_img,
        }

        logits = self.model_txt(captions, **model_kwargs).logits
        full_log_probs = logits.log_softmax(-1)[:, self.prefix_len - 1 : -1]

        entropies = full_log_probs.exp() * full_log_probs
        entropies = -entropies.sum(-1)
        entropies *= mask.to(entropies.dtype)
        entropies = entropies.sum(-1)
        entropies /= msg_lengths.to(entropies.dtype)

        log_probs = full_log_probs
        log_probs = log_probs.gather(
            2, captions[:, self.prefix_len :].unsqueeze(2)
        ).squeeze(-1)
        log_probs *= mask.to(log_probs.dtype)
        log_probs = log_probs.sum(-1) / msg_lengths.to(log_probs.dtype)

        captions = captions[:, self.prefix_len :]
        decoded_captions = self.tokenizer.batch_decode(
            captions,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        return decoded_captions, log_probs, entropies, msg_lengths
