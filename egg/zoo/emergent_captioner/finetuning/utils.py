# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from typing import Any, Dict, NamedTuple, Optional
import os
import yaml
import torch
from transformers import GPT2LMHeadModel, LogitsProcessor, get_linear_schedule_with_warmup
from egg.core import Callback, Interaction
import wandb

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
        # print(f"------->{self.lm.device}")
        # for i in self.lm.named_parameters():
        #     print(f"{i[0]} -> {i[1].device}")
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
        self.is_ddp = self.trainer.distributed_context.is_distributed

        optimizer_schedule_state_dict = None
        if self.trainer.optimizer_scheduler:
            optimizer_schedule_state_dict = (
                self.trainer.optimizer_scheduler.state_dict()
            )

        # if self.is_ddp:
        #     game = self.trainer.game.module
        #     self.trainer.game.module.loss.remove_fields_negatives()
        #     self.trainer.game.module.sender.unpatch_model()

        # else:
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

    def save_clipclap_model(self, epoch=None, model_name = None, SAVE_BEST_METRIC = None ):
        self.is_ddp = self.trainer.distributed_context.is_distributed

        if hasattr(self.trainer, "checkpoint_path"):
            if (
                self.trainer.checkpoint_path
                and self.trainer.distributed_context.is_leader
            ):
                self.trainer.checkpoint_path.mkdir(exist_ok=True, parents=True)
                if model_name is None :
                    model_name = f"clip_cap_model_e_{epoch if epoch else 'final'}.pt"
                else:
                    model_name  = f"e_{epoch if epoch else 'final'}.pt"
                
                if SAVE_BEST_METRIC:
                    model_name = f"best.pt"
                
                torch.save(
                    self.get_checkpoint(),
                    self.trainer.checkpoint_path / model_name,
                )
                # if self.is_ddp:
                #     self.trainer.game.module.sender.patch_model()
                # else:                    
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
            self.save_clipclap_model(model_name = model_name)

def get_config(filename):
    config_path = os.path.join(os.getcwd(),filename)
    with open(config_path) as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    return config

def process_config(config, use_ddp):
    """
    /home/manugaur —> /ssd_scratch/cvit/manu
    """
    ssd_scratch = "/ssd_scratch/cvit/manu"
    jatayu  = os.path.isdir("/home/manugaur")
    if not jatayu:
        config['opts']['dataset_dir'] = os.path.join(ssd_scratch, config['opts']['dataset_dir'].split("manugaur/")[-1])
        config['opts']['mle_model_path'] = os.path.join(ssd_scratch, config['opts']['mle_model_path'].split("manugaur/")[-1])
        config['opts']['checkpoint_dir'] = os.path.join(ssd_scratch,config['opts']['checkpoint_dir'].split("manugaur/")[-1])
        config['opts']['jatayu'] = jatayu
        config["official_clipcap_weights"] = os.path.join(ssd_scratch, config["official_clipcap_weights"].split("manugaur/")[-1])
    
    if use_ddp:
        config["num_workers"] = 0
    return config

def get_cl_args(config):
    params = []
    for k,v in config['opts'].items():
        params.append(f"--{k}")
        params.append(f"{v}")
    return params

def init_wandb(config):
    if config['WANDB']['logging'] and (not config['WANDB']['sweep']) :
        wandb.init(entity= config["WANDB"]["entity"], project=config["WANDB"]['project'], config = config)
        wandb.run.name = config['WANDB']['run_name']

def get_best_state_dict(config):
    print("| LOADED BEST MODEL FOR INFERENCE")
    
    desired_format_state_dict = torch.load(config["official_clipcap_weights"])
    saved_state_dict = torch.load(os.path.join(config["opts"]["checkpoint_dir"], "best.pt"))[1]
    state_dict = {}
    for idx, k in enumerate(desired_format_state_dict.keys()):
        state_dict[k] = saved_state_dict["sender.clipcap." + k]
    return state_dict
