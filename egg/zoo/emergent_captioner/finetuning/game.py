# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable
import wandb
import torch
import torch.nn as nn

from egg.core.baselines import MeanBaseline, NoBaseline
from egg.core.interaction import LoggingStrategy
from egg.zoo.emergent_captioner.finetuning.blip import BlipSender
from egg.zoo.emergent_captioner.finetuning.clipcap import ClipCapSender
from egg.zoo.emergent_captioner.finetuning.losses import get_loss
from egg.zoo.emergent_captioner.finetuning.receiver import ClipReceiver


class ReinforceCaptionGame(nn.Module):
    def __init__(
        self,
        sender: nn.Module,
        receiver: nn.Module,
        loss: Callable,
        kl_div_coeff: float = 0.0,
        baseline: str = "no",
        train_logging_strategy: LoggingStrategy = None,
        test_logging_strategy: LoggingStrategy = None,
    ):
        super(ReinforceCaptionGame, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.loss = loss

        self.baseline_name = baseline
        self.baseline = {"no": NoBaseline, "mean": MeanBaseline}[baseline]()

        self.kl_div_coeff = kl_div_coeff

        self.train_logging_strategy = (
            LoggingStrategy().minimal()
            if train_logging_strategy is None
            else train_logging_strategy
        )
        self.test_logging_strategy = (
            LoggingStrategy().minimal()
            if test_logging_strategy is None
            else test_logging_strategy
        )

    def forward(self, sender_input, labels, receiver_input=None, aux_input=None):
        """
        receiver : clip
        sender : clip VIT + clipcap model
        kl_div_coeff : 0
        _____
        sender_input : (B, 3, 224, 224)
        labels : ? 
        receiver_input : (B, 3 ,224, 224)
        aux : dict {img_id : cocoids
                    captions : 5 coco GT cap for each image : list of 5 lists --> each sublist has bsz captions
                    }
        """
        captions, log_prob, kl_div = self.sender(sender_input, aux_input) # logprob : (B) --> only one logprob per caption (averaged over all words)

        with torch.no_grad():
            text_feats, img_feats = self.receiver(captions, receiver_input, aux_input) #clip_feats
            loss, aux_info = self.loss(text_feats, img_feats, labels, aux_input)
        weighted_kl_div = self.kl_div_coeff * kl_div
        
        baseline = self.baseline.predict(loss.detach())

        reward = (loss.detach() - baseline) + weighted_kl_div
        policy_loss = (reward * log_prob).mean()
        if self.training:
            self.baseline.update(loss)

        aux_info["kl_div"] = kl_div

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )
        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input,
            labels=labels,
            receiver_input=receiver_input,
            aux_input=aux_input,
            message=captions,
            receiver_output=None,
            message_length=None,
            aux=aux_info,
        )

        return policy_loss.mean(), interaction, reward.mean().item()


def build_game(opts):
    if opts.captioner_model.lower() == "clipcap":
        sender = ClipCapSender(
            clip_model=opts.sender_clip_model,
            clipcap_path=opts.clipcap_model_path,
            do_sample=opts.do_sample,
            beam_size=opts.beam_size,
            max_len=opts.max_len,
        )
    elif opts.captioner_model.lower() == "blip":
        sender = BlipSender(
            blip_model=opts.blip_model,
            beam_size=opts.beam_size,
            max_len=opts.max_len,
            freeze_visual_encoder=opts.freeze_blip_visual_encoder,
        )
    else:
        raise RuntimeError

    receiver = ClipReceiver(clip_model=opts.recv_clip_model)

    test_logging_strategy = LoggingStrategy(
        False, False, True, True, True, False, False
    )

    loss_fn = get_loss(
        loss_type=opts.loss_type,
        dataset=opts.train_dataset,
        num_hard_negatives=opts.num_hard_negatives,
    )
    # remember that with non-diff losses you should use a wrapper around recv
    game = ReinforceCaptionGame(
        sender=sender,
        receiver=receiver,
        loss=loss_fn,
        baseline=opts.baseline,
        kl_div_coeff=opts.kl_div_coeff,
        test_logging_strategy=test_logging_strategy,
    )
    return game
