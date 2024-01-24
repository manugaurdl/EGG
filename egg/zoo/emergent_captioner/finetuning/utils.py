# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from typing import Any, Dict, NamedTuple, Optional
import os
import yaml
import torch
from transformers import GPT2LMHeadModel, LogitsProcessor

from egg.core import Callback, Interaction


class MyCheckpoint(NamedTuple):
    epoch: int
    model_state_dict: Dict[str, Any]
    optimizer_state_dict: Dict[str, Any]
    optimizer_scheduler_state_dict: Optional[Dict[str, Any]]
    opts: argparse.ArgumentParser


class KLRegularizer:
    def __init__(self, device=torch.device("cuda")):
        self.lm = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

    @torch.no_grad()
    def compute_kl_loss(self, indices, log_probs):
        # 50256 is gpt2 beginning of sentence
        indices = torch.cat([torch.ones_like(indices[:, :1]) * 50256, indices], dim=1)
        # we take probs from bos until last token
        generated = self.lm(indices)["logits"].log_softmax(-1)[:, :-1, :]

        step_kl_div = []
        for timestep in range(generated.shape[1]):
            x = torch.nn.functional.kl_div(
                log_probs[:, timestep],
                generated[:, timestep],
                log_target=True,
                reduction="none",
            )
            step_kl_div.append(x.sum(-1))  # summing over vocab_dim
        kl_div = torch.stack(step_kl_div, dim=1)
        return kl_div


class StopTokenLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, do_sample):
        self.eos_token_id = tokenizer.eos_token_id #50256

        # f : tokenizer.convert_ids_to_tokens --> decodes token  i.e f(500) = "walk" 
        # There are 121 strings that contain "." --> all of them are treated as stop tokens
        self.stop_word_ids = set(
            [
                idx
                for idx in range(len(tokenizer))
                if "." in tokenizer.convert_ids_to_tokens(idx)
            ]
        )
        self.vocab_size = len(tokenizer)

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        # iterate each batch of prefix tokens ; input_ids (B , 10)
        for i, input_id in enumerate(input_ids):
            if input_id[-1].item() in self.stop_word_ids:
                scores[i, : self.vocab_size] = torch.finfo().min
                scores[i, self.vocab_size :] = float("-inf")
                scores[i, self.eos_token_id] = 0.0
        return scores


class ModelSaver(Callback):
    def __init__(self, opts: argparse.ArgumentParser):
        self.opts = opts

    def get_checkpoint(self):
        optimizer_schedule_state_dict = None
        if self.trainer.optimizer_scheduler:
            optimizer_schedule_state_dict = (
                self.trainer.optimizer_scheduler.state_dict()
            )
        if self.trainer.distributed_context.is_distributed:
            game = self.trainer.game.module
        else:
            game = self.trainer.game
        # cleaning a model such that it has default settings e.g. no buffer and no modules/tensors in the loss
        # this is done to avoid mandatory fields when loading a model e.g. a tensor of negatives
        self.trainer.game.loss.remove_fields_negatives()
        self.trainer.game.sender.unpatch_model()
        return MyCheckpoint(
            epoch=self.epoch,
            model_state_dict=game.state_dict(),
            optimizer_state_dict=self.trainer.optimizer.state_dict(),
            optimizer_scheduler_state_dict=optimizer_schedule_state_dict,
            opts=self.opts,
        )

    def save_clipclap_model(self, epoch="", model_name = None, SAVE_BEST_METRIC = None ):
        # print(bdfaefndsjkfhnasdknfjk)
        if hasattr(self.trainer, "checkpoint_path"):
            if (
                self.trainer.checkpoint_path
                and self.trainer.distributed_context.is_leader
            ):
                self.trainer.checkpoint_path.mkdir(exist_ok=True, parents=True)
                if model_name is None :
                    model_name = f"clip_cap_model_e_{epoch if epoch else 'final'}.pt"
                else:
                    model_name  = f"{model_name}_e_{epoch if epoch else 'final'}.pt"
                
                if SAVE_BEST_METRIC:
                    model_name = f"{model_name.split('_e_')[0]}_best.pt"
                
                torch.save(
                    self.get_checkpoint(),
                    self.trainer.checkpoint_path / model_name,
                )
                self.trainer.game.sender.patch_model()

    def on_epoch_end(self, loss: float, _logs: Interaction, epoch: int, model_name : str, SAVE_BEST_METRIC: bool):
        self.epoch = epoch
        if self.opts.captioner_model == "clipcap":
            self.save_clipclap_model(epoch=epoch, model_name = model_name, SAVE_BEST_METRIC = SAVE_BEST_METRIC)

    def on_train_end(self, epoch : int, model_name : str):

        try:
            isinstance(self.epoch, ModelSaver)
        except:
            self.epoch = epoch
        
        if self.opts.captioner_model == "clipcap":
            self.save_clipclap_model(epoch = self.epoch, model_name = model_name)

def get_config():
    config_path = os.path.join(os.getcwd(),'egg/zoo/emergent_captioner/finetuning/config.yml')
    with open(config_path) as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    return config

def set_data_dir(config):
    """
    /home/manugaur —> /ssd_scratch/cvit/manu
    """
    jatayu  = os.path.isdir("/home/manugaur")
    if not jatayu:
        config['opts']['dataset_dir'] = os.path.join("/ssd_scratch/cvit/manu",config['opts']['dataset_dir'].split("manugaur/")[-1])
        config['opts']['clipcap_model_path'] = os.path.join("/ssd_scratch/cvit/manu",config['opts']['clipcap_model_path'].split("manugaur/")[-1])
        config['opts']['checkpoint_dir'] = os.path.join("/ssd_scratch/cvit/manu",config['opts']['checkpoint_dir'].split("manugaur/")[-1])
        config['opts']['jatayu'] = jatayu
    return config

def get_cl_args(config):
    params = []
    for k,v in config['opts'].items():
        params.append(f"--{k}")
        params.append(f"{v}")
    return params