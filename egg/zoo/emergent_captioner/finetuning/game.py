# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from egg.core.baselines import MeanBaseline, NoBaseline
from egg.core.interaction import LoggingStrategy
from egg.zoo.emergent_captioner.finetuning.blip import BlipSender
from egg.zoo.emergent_captioner.finetuning.clipcap import ClipCapSender
from egg.zoo.emergent_captioner.finetuning.losses import get_loss, CiderReward, DiscriminativeLoss
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
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


    def forward(self, sender_input, labels,receiver_input=None, aux_input=None, GREEDY_BASELINE = False, train_method= None, prefix_len = 10):
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
        CIDER_OPTIM = isinstance(self.loss, CiderReward)
        if CIDER_OPTIM:
            # get policy cap
            policy_captions, log_prob, kl_div = self.sender(sender_input, aux_input, CIDER_OPTIM) # logprob : (B) --> only one logprob per caption (averaged over all words)
        
            #get greedy_cap
            if GREEDY_BASELINE:
                self.eval()
                with torch.no_grad():
                    greedy_cap, _, _  = self.sender(sender_input, aux_input, False, GREEDY_BASELINE)
                    baseline = torch.tensor(self.loss(greedy_cap, aux_input)).to(log_prob.device).detach()
                self.train()

            # gt : aux_input
            # baseline --> mean vs greedy
            policy_cider = torch.tensor(self.loss(policy_captions, aux_input)).to(log_prob.device)
                        
            weighted_kl_div = self.kl_div_coeff * kl_div
            
            if not GREEDY_BASELINE:
                # get policy cap first.
                baseline = self.baseline.predict(policy_cider.detach())

            reward = (policy_cider.detach() - baseline) + weighted_kl_div
            reinforce_loss = -1*((reward * log_prob).mean())

            # import ipdb;ipdb.set_trace()
            if self.training and not GREEDY_BASELINE:
                self.baseline.update(policy_cider)

            aux_info = {'acc' : torch.randn(1,2), "kl_div" : kl_div}
            
            logging_strategy = self.test_logging_strategy

            interaction = logging_strategy.filtered_interaction(
                sender_input=sender_input,
                labels=labels,
                receiver_input=receiver_input,
                aux_input=aux_input,
                message=policy_captions,
                receiver_output=None,
                message_length=None,
                aux=aux_info,
                )
        
        elif isinstance(self.loss, DiscriminativeLoss):
            captions, log_prob, kl_div = self.sender(sender_input, aux_input) # logprob : (B) --> only one logprob per caption (averaged over all words)

            with torch.no_grad():
                text_feats, img_feats = self.receiver(captions, receiver_input, aux_input) #clip_feats
                loss, aux_info = self.loss(text_feats, img_feats, labels, aux_input)
            weighted_kl_div = self.kl_div_coeff * kl_div
            
            baseline = self.baseline.predict(loss.detach())

            reward = (loss.detach() - baseline) + weighted_kl_div
            reinforce_loss = (reward * log_prob).mean()
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

        else:
            outputs = self.sender(sender_input, aux_input, train_method= train_method)
            if self.training:
                targets = aux_input['tokens'].view(-1, aux_input["tokens"].shape[-1])
                mask = aux_input['mask'].view(-1, aux_input["mask"].shape[-1])
                # targets, mask = targets.to(device), mask.to(device)
                logits = outputs.logits[:, prefix_len - 1: -1]
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.to(torch.long).flatten(), ignore_index=0) # (B,T) flattened to (B*T)
                # probs = torch.nn.functional.softmax(logits, dim=-1)
                # preds = torch.multinomial(probs.view(-1, probs.shape[-1]), num_samples=1) # preds is flattened out --> (B*max_cap_len , 1)
                baseline = self.baseline.predict(loss.detach())
                self.baseline.update(loss)
            else:
                val_captions, log_prob, kl_div = outputs
                aux_info = {"kl_div" :kl_div}

                logging_strategy = (
                    self.train_logging_strategy if self.training else self.test_logging_strategy
                )
                interaction = logging_strategy.filtered_interaction(
                    sender_input=sender_input,
                    labels=labels,
                    receiver_input=receiver_input,
                    aux_input=aux_input,
                    message=val_captions,
                    receiver_output=None,
                    message_length=None,
                    aux=aux_info,
                )
                return torch.randn(1), interaction, torch.randn(1)




            # weighted_kl_div = self.kl_div_coeff * kl_div
            # aux_info["kl_div"] = weighted_kl_div
            aux_info = {}
            captions = []
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
            
            return loss, interaction, loss.item()

        return reinforce_loss.mean(), interaction, reward.mean().item()


def build_game(opts, config):
    if opts.captioner_model.lower() == "clipcap":
        sender = ClipCapSender(
            clip_model=opts.sender_clip_model,
            clipcap_path=opts.mle_model_path,
            official_clipcap_weights = config["official_clipcap_weights"],
            train_method= config["train_method"],
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
