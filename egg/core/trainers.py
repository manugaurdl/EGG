# Global vars
STEP = 0

import os
import wandb
import pathlib
import pickle
import json
import numpy as np
from typing import List, Optional
from prettytable import PrettyTable
from tqdm import tqdm
import torch.distributed as dist

try:
    # requires python >= 3.7
    from contextlib import nullcontext
except ImportError:
    # not exactly the same, but will do for our purposes
    from contextlib import suppress as nullcontext

import torch
from torch.utils.data import DataLoader

from .batch import Batch
from .callbacks import (
    Callback,
    Checkpoint,
    CheckpointSaver,
    ConsoleLogger,
    TensorboardLogger,
)
from .distributed import get_preemptive_checkpoint_dir
from .interaction import Interaction
from .util import get_opts, move_to
from egg.zoo.emergent_captioner.utils import (
    dump_interaction,
    get_sha,
    log_stats,
    print_grad_info,
    setup_for_distributed,
    store_job_and_task_id,
)
from egg.zoo.emergent_captioner.evaluation.evaluate_nlg import compute_nlg_metrics
from egg.zoo.emergent_captioner.finetuning.losses import DiscriminativeLoss
from egg.zoo.emergent_captioner.evaluation.mmvp_vlm import mmvp_vlm_benchmark


try:
    from torch.cuda.amp import GradScaler, autocast
except ImportError:
    pass

def int2mil(number):
    if abs(number) >= 100_000:
        formatted_number = "{:.1f}M".format(number / 1_000_000)
    else:
        formatted_number = str(number)
    return formatted_number

def trainable_params(model):
    # print(f'{int2mil(sum(p.numel() for p in model.parameters() if p.requires_grad == True))} trainable params')
    return int2mil(sum(p.numel() for p in model.parameters() if p.requires_grad == True))
    # return sum(p.numel() for p in model.parameters() if p.requires_grad == True)

def get_loader(epoch, ranges):
    prev_end = 0
    sorted_items = sorted(ranges.items(), key=lambda x: x[1])

    for key, end in sorted_items:
        if prev_end <= epoch <end:
            return key
        prev_end = end
    raise Exception("epoch out of curricullum")

def count_trainable_parameters(model):
    table = PrettyTable(["Modules", "Requires grad", "Trainable parameters"])
    table.align["Modules"] = "l"
    table.align["Requires grad"] = "c"
    table.align["Trainable parameters"] = "r"

    total_params = 0
    for name, parameter in model.named_parameters():
        
        requires_grad = True
        params = parameter.numel()
        
        if not parameter.requires_grad:
            requires_grad = False
            params = 0
        
        table.add_row([name, requires_grad, params])
        total_params += params

    print(table)
    print(f"Total Trainable Params: {total_params:,}")
    return total_params

    
class Trainer:
    """
    Implements the training logic. Some common configuration (checkpointing frequency, path, validation frequency)
    is done by checking util.common_opts that is set via the CL.
    """

    def __init__(
        self,
        game: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loaders: dict,
        optimizer_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        validation_data_rand: Optional[DataLoader] = None,
        validation_data_neg: Optional[DataLoader] = None,
        inference_data : Optional[DataLoader] = None,
        device: torch.device = None,
        callbacks: Optional[List[Callback]] = None,
        grad_norm: float = None,
        aggregate_interaction_logs: bool = True,
        debug: bool = False,
    ):
        """
        :param game: A nn.Module that implements forward(); it is expected that forward returns a tuple of (loss, d),
            where loss is differentiable loss to be minimized and d is a dictionary (potentially empty) with auxiliary
            metrics that would be aggregated and reported
        :param optimizer: An instance of torch.optim.Optimizer
        :param optimizer_scheduler: An optimizer scheduler to adjust lr throughout training
        :param train_data: A DataLoader for the training set
        :param validation_data: A DataLoader for the validation set (can be None)
        :param device: A torch.device on which to tensors should be stored
        :param callbacks: A list of egg.core.Callback objects that can encapsulate monitoring or checkpointing
        """
        self.game = game
        self.optimizer = optimizer
        self.optimizer_scheduler = optimizer_scheduler
        self.train_loaders =  train_loaders
        self.val_loader_rand = validation_data_rand
        self.val_loader_neg = validation_data_neg
        self.inference_loader = inference_data 
        common_opts = get_opts()
        self.validation_freq = common_opts.validation_freq
        self.device = common_opts.device if device is None else device
        self.debug = debug

        self.should_stop = False
        self.start_epoch = 0  # Can be overwritten by checkpoint loader
        self.callbacks = callbacks if callbacks else []
        self.grad_norm = grad_norm
        self.aggregate_interaction_logs = aggregate_interaction_logs

        self.update_freq = common_opts.update_freq

        if common_opts.load_from_checkpoint is not None:
            print(
                f"# Initializing model, trainer, and optimizer from {common_opts.load_from_checkpoint}"
            )
            self.load_from_checkpoint(common_opts.load_from_checkpoint)

        self.distributed_context = common_opts.distributed_context
        if self.distributed_context.is_distributed:
            print("# Distributed context: ", self.distributed_context)

        if self.distributed_context.is_leader and not any(
            isinstance(x, CheckpointSaver) for x in self.callbacks
        ):
            if common_opts.preemptable:
                assert (
                    common_opts.checkpoint_dir
                ), "checkpointing directory has to be specified"
                d = get_preemptive_checkpoint_dir(common_opts.checkpoint_dir)
                self.checkpoint_path = d
                self.load_from_latest(d)
            else:
                self.checkpoint_path = (
                    None
                    if common_opts.checkpoint_dir is None
                    else pathlib.Path(common_opts.checkpoint_dir)
                )

            if self.checkpoint_path:
                checkpointer = CheckpointSaver(
                    checkpoint_path=self.checkpoint_path,
                    checkpoint_freq=common_opts.checkpoint_freq,
                )
                self.callbacks.append(checkpointer)

        if self.distributed_context.is_leader and common_opts.tensorboard:
            assert (
                common_opts.tensorboard_dir
            ), "tensorboard directory has to be specified"
            tensorboard_logger = TensorboardLogger()
            self.callbacks.append(tensorboard_logger)

        if self.callbacks is None:
            self.callbacks = [
                ConsoleLogger(print_train_loss=False, as_json=False),
            ]

        if self.distributed_context.is_distributed:
            print(f"Wrapping model to GPU:{self.distributed_context.local_rank}")
            device_id = self.distributed_context.local_rank
            torch.cuda.set_device(device_id)
            torch.cuda.empty_cache()
            self.game.to(device_id)

            # NB: here we are doing something that is a bit shady:
            # 1/ optimizer was created outside of the Trainer instance, so we don't really know
            #    what parameters it optimizes. If it holds something what is not within the Game instance
            #    then it will not participate in distributed training
            # 2/ if optimizer only holds a subset of Game parameters, it works, but somewhat non-documentedly.
            #    In fact, optimizer would hold parameters of non-DistributedDataParallel version of the Game. The
            #    forward/backward calls, however, would happen on the DistributedDataParallel wrapper.
            #    This wrapper would sync gradients of the underlying tensors - which are the ones that optimizer
            #    holds itself.  As a result it seems to work, but only because DDP doesn't take any tensor ownership.

            self.game = torch.nn.parallel.DistributedDataParallel(
                self.game,
                device_ids=[device_id],
                output_device=device_id,
                find_unused_parameters=True,
            )
            self.optimizer.state = move_to(self.optimizer.state, device_id)

        else:
            self.game.to(self.device)
            # NB: some optimizers pre-allocate buffers before actually doing any steps
            # since model is placed on GPU within Trainer, this leads to having optimizer's state and model parameters
            # on different devices. Here, we protect from that by moving optimizer's internal state to the proper device
            self.optimizer.state = move_to(self.optimizer.state, self.device)

        if common_opts.fp16:
            self.scaler = GradScaler()
        else:
            self.scaler = None

    def eval(self, loader, inference : bool, config : dict, data=None, GREEDY_BASELINE = False, train_method = None):

        global STEP
        mean_loss = 0.0
        interactions = []
        n_batches = 0
        self.game.eval()

        with torch.no_grad():
            for batch_id, batch in tqdm(enumerate(loader), total = len(loader)):
                if not isinstance(batch, Batch):
                    batch = Batch(*batch)
                batch = batch.to(self.device)
                
                optimized_loss, interaction, reward = self.game(*batch, train_method = train_method, inference=inference)

                
                """
                interaction : sender_input=None, receiver_input=None, labels = tensor, aux_input = {cocoid, captions, tokens, mask}, message, receiver_output=None, message_length=None, aux = {"kl_div" = torch.rand(1)}
                """
                # lst = []
                # for _ in range(self.distributed_context.world_size):
                #     lst.append(torch.zeros_like(interaction.labels))

                # dist.all_gather(lst, interaction.labels)
                # interaction = torch.cat(lst, dim=0).to("cpu")
                # torch.save(interaction, "/ssd_scratch/cvit/manu/temp_interaction.pt")

                interaction = interaction.to("cpu")
                mean_loss += optimized_loss

                for callback in self.callbacks:
                    callback.on_batch_end(
                        interaction, optimized_loss, n_batches, is_training=False
                    )
                if interaction.sender_input is None:
                    interaction.sender_input = torch.rand((3,1)).float()
                
                if inference or isinstance(self.game.loss, DiscriminativeLoss):
                    if interaction.receiver_input is None:
                        interaction.receiver_input = torch.rand((3,1)).float()
                    #only for SR loss --> during SR training or MLE/CiDEr inference
                    interaction.aux["mean_rank"] = interaction.aux["mean_rank"].view(1,1)
                    interaction.aux["median_rank"] = interaction.aux["median_rank"].view(1,1) 
            
                interactions.append(interaction)
                n_batches += 1


        mean_loss /= n_batches
        #if data is dict/tensor --> its gets extended N_batch times. If its a list, a new list of list gets created of len = N_batch
        full_interaction = Interaction.from_iterable(interactions)
        img_ids = full_interaction.aux_input['cocoid']
        captions = full_interaction.aux_input['captions']
        preds_per_batch = full_interaction.message
        bsz = len(preds_per_batch[0])
        gold_standard = {}
        
        for i, batch in enumerate(captions):
            coco_caps = list(zip(*batch))
            for j, img in enumerate(coco_caps):
                gold_standard[(i)*bsz + j] = [{"caption": cap} for cap in img]
        
        predictions = {}
        
        for i, batch in enumerate(preds_per_batch):
            for j, pred in enumerate(batch):
                predictions[(i*bsz) + j] = [{"caption" :pred}]
        
        summary = compute_nlg_metrics(predictions, gold_standard) # score for each idx stored in summary except bleu


        # MMVP eval
        if config['finetune_model'] == "clip":
            mmvp_results = mmvp_vlm_benchmark(self.game.sender.clip, self.game.sender.clip_preproc, "/home/manugaur/MMVP/mmvp_vlm")
            full_interaction.aux["mmvp_avg"] =  np.array(list(mmvp_results.values())).mean()
            full_interaction.aux.update({"mmvp_all" : mmvp_results})

        return mean_loss.item(), full_interaction, reward, summary

    def train_epoch(self,loader, WANDB, GREEDY_BASELINE, train_method, opts, config):

        global STEP
        mean_loss = 0
        n_batches = 0
        interactions = []
        
        self.game.train()
        if config["finetune_model"]=="gpt":
            self.game.sender.clip.eval()
        else:
            self.game.sender.clip.train()
            self.game.sender.clipcap.gpt.eval()
        self.game.receiver.eval()
        self.optimizer.zero_grad()

        for batch_id, batch in tqdm(enumerate(loader), total = len(loader)):
            # batch.append(GREEDY_BASELINE)
            if self.debug and batch_id == 10:
                break
            if not isinstance(batch, Batch):
                batch = Batch(*batch)
            batch = batch.to(self.device)

            context = autocast() if self.scaler else nullcontext()
            with context:
                optimized_loss, interaction, reward = self.game(*batch, GREEDY_BASELINE, train_method)
                
                #not accumulating gradients currently
                if self.update_freq > 1:
                    # throughout EGG, we minimize _mean_ loss, not sum
                    # hence, we need to account for that when aggregating grads
                    optimized_loss = optimized_loss / self.update_freq


            if self.scaler:
                self.scaler.scale(optimized_loss).backward()
            else:
                optimized_loss.backward()

            if batch_id % self.update_freq == self.update_freq - 1:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)

                if self.grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.game.parameters(), self.grad_norm
                    )
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

            n_batches += 1
            mean_loss += optimized_loss.detach()
            # print("before gather")
            # print(interaction.aux["acc"])

            # if (self.distributed_context.is_distributed and self.aggregate_interaction_logs):
            #     interaction = Interaction.gather_distributed_interactions(interaction)
            
            # print("after gather")
            # print(interaction.aux["acc"])

            interaction = interaction.to("cpu")

            for callback in self.callbacks:
                callback.on_batch_end(interaction, optimized_loss, batch_id)

            interactions.append(interaction)
            print(f"Loss : {optimized_loss.item():.5f}")
            print(f"Avg Loss : {(mean_loss.item())/n_batches:.5f}")
            train_log = { "Loss" :optimized_loss.item(),
                            "Reward" : reward,
                            "train R@1" : interaction.aux['batch_acc1'].item(),
                            "lr" : self.optimizer.state_dict()["param_groups"][0]["lr"]
                            }
            if train_method != "mle":
                train_log["log_prob"] = interaction.aux['log_prob'].mean().item()

            if opts.loss_type != 'cider':
                train_log["acc@1"] : interaction.aux['acc'].mean().item()

            if WANDB:
                wandb.log(train_log, step = STEP)
            STEP+=1
            if self.optimizer_scheduler:
                self.optimizer_scheduler.step()

        mean_loss /= n_batches
        full_interaction = Interaction.from_iterable(interactions)

        # print("****"*30)
        # # print(sum(p.mean() for p in self.game.sender.clip.visual.parameters()))
    
        # new_weights = {}
        # for name, param in self.game.sender.clipcap.gpt.transformer.named_parameters():
        #     new_weights[name] = param
        # # print(new_weights["h.0.attn.c_attn.parametrizations.weight.original"].requires_grad)
        # # print(f"og weight = {new_weights['h.0.attn.c_attn.parametrizations.weight.original'].mean()}")
        # print(f"lora weight = {new_weights['h.0.attn.c_proj.parametrizations.weight.0.lora_A'].mean()}")

        # print(f"LoRA param : {sum(p.mean() for p in self.game.sender.clipcap.gpt.transformer.h[0].attn.c_attn.parameters())}")
        # print(f"frozen param : {next(self.game.sender.clipcap.gpt.lm_head.parameters()).sum()}")
        # print(next(self.game.sender.clipcap.gpt.lm_head.parameters()).requires_grad)

        # print("****"*30)
        return mean_loss.item(), full_interaction

    def train(self,config, opts, inference = False):


        print(f"Total trainable params : {trainable_params(self.game.sender)}")
        count_trainable_parameters(self.game.sender)
        global STEP        
        n_epochs = config['opts']['n_epochs']
        WANDB = config['WANDB']['logging']
        if self.distributed_context.is_distributed:
            WANDB = WANDB and self.distributed_context.is_leader
        print(f"+++++++ WANDB  ={WANDB} LOCAL_RANK = {self.distributed_context.is_leader}  ++++++")
        INIT_VAL = config['INIT_VAL']
        GREEDY_BASELINE = config['GREEDY_BASELINE']
        SAVE_BEST_METRIC = config['SAVE_BEST_METRIC']
        train_method = config["train_method"]
        SAVE_USING_NEG = config["neg_mining"]["save_using_neg"] and config["neg_mining"]["do"]  
        best_metric_score = 0

        def prepare_logs(summary, loss, interaction, reward, train_method, loss_type, epoch, config):

            val_log = { "Val Loss" : loss,
                        "Val Reward" : reward,
                        "CIDEr" : summary["CIDEr"],
                        "SPICE" : summary["SPICE"],
                        "Bleu_4" : summary["Bleu_4"],
                        'METEOR': summary["METEOR"],
                        "ROUGE_L" : summary['ROUGE_L'],
                        }
            if config["finetune_model"] == "clip":
                val_log["mmvp_avg"] = interaction.aux['mmvp_avg']
                
                if config["WANDB"]["log_mmvp_all"]:
                    val_log.update(interaction.aux['mmvp_all'])
            # if WANDB.log _mmvp_aspects:
            #     val_log.update(all mmvp aspects from interaction.aux)

            if train_method == "mle":
                del val_log["Val Loss"]
                del val_log["Val Reward"]

            if loss_type == 'discriminative':
                metric =  interaction.aux['acc'].mean().item()
                val_log["VAL_R@1"] = metric
            
            else:
                metric = summary["CIDEr"]            

            # aggregated print for 1st obj, pass for other 2
            for callback in self.callbacks:
                callback.on_validation_end(loss, interaction, epoch + 1)

            return val_log, metric

        def run_validation(loader, epoch : int, config : dict,  inference : bool = False):
            validation_loss = validation_interaction = None
            if (
                loader is not None
                and self.validation_freq > 0
                and (epoch + 1) % self.validation_freq == 0
            ):
                # for idx, callback in enumerate(self.callbacks): 
                #     if idx in [0,1]:
                #         continue
                #     callback.on_validation_begin(epoch + 1) # pass
                validation_loss, validation_interaction, val_reward, summary = self.eval(loader, inference = inference, config = config, GREEDY_BASELINE = GREEDY_BASELINE, train_method = train_method)

                val_log, metric = prepare_logs(summary, validation_loss, validation_interaction, val_reward, train_method, opts.loss_type, epoch, config)

                return val_log, validation_interaction, metric

        def log(log, interaction, metric, epoch, name, config = None, inference=False):
                """val log is plotted on wandb. Inference log is saved as json."""

                if inference:
                    #save inference preds
                    self.save_val_preds(interaction, config, inference = True)
                    
                    # save inference log
                    test_log = {}
                    test_log['recall_1'] = interaction.aux['acc'].mean().item()
                    test_log['recall_5'] = interaction.aux['acc_5'].mean().item()
                    test_log['CLIP_s'] = interaction.aux['clip_s'].mean().item()
                    test_log.update(log)

        
                    with open(os.path.join(inference_log_dir,  f"{config['captions_type']}_{config['opts']['checkpoint_dir'].split('/')[-1]}.json"), "w") as f:
                        json.dump(test_log, f)    
                    # with open("/home/manugaur/EGG/inference_log/blip2mistral_mle.json", "w") as f:
                    #     json.dump(test_log, f)

                else:

                    log["epoch"] = epoch
                    log["val_log_prob"] =  interaction.aux['log_prob'].mean().item()

                    if name == "rand":
                        wandb.log(log, step = STEP)
                    else:
                        wandb.log({"VAL_R@1_NEG" : log["VAL_R@1"]}, step = STEP)



        def rand_neg_val(epoch : int, WANDB : bool,  config : dict, inference : bool = False):

            if inference:
                test_log, interaction, metric = run_validation(self.inference_loader, epoch, config, inference)
                log(test_log, interaction, metric, None, None, config = config, inference = inference)
            
            else:
                rand_log, rand_interaction, metric = run_validation(self.val_loader_rand, epoch, config, inference)
                if self.val_loader_neg is not None:
                    neg_log, neg_interaction, neg_metric = run_validation(self.val_loader_neg, epoch, config, inference)

                if WANDB:
                    log(rand_log, rand_interaction, metric, epoch, "rand", config = config)
                    
                    if self.val_loader_neg is not None:
                        log(neg_log, neg_interaction, neg_metric, epoch, "neg", config = config)
                
                if SAVE_USING_NEG and self.val_loader_neg is not None:
                    return neg_metric

            torch.cuda.empty_cache()
            return metric


        inference_log_dir = os.path.join(config["inference"]["output_dir"].split("/inference")[0], "inference_log")
        if not os.path.isdir(inference_log_dir):
            os.makedirs(inference_log_dir)

        #INIT VAL
        if inference or (INIT_VAL and self.distributed_context.is_leader):
            metric = rand_neg_val(0, WANDB, config = config,  inference=inference)
        if inference:                
            return

        for callback in self.callbacks:
            """
            In CallBack class, create self.trainer = callbacks.console_logger , finetuning.utils.ModelSaver , callbacks.checkpointsaver
            """
            callback.on_train_begin(self)

        
        for epoch in range(self.start_epoch, n_epochs):

            if self.distributed_context.is_distributed:
                self.train_data.sampler.set_epoch(epoch)
                # self.validation_data.sampler.set_epoch(epoch)     

            # Train epoch
            if self.distributed_context.is_distributed:
                dist.barrier()
            print(f"Training epoch {epoch + 1}")

            # for callback in self.callbacks:
            #     callback.on_epoch_begin(epoch + 1)                 
            loader = get_loader(epoch, config['neg_mining']['curricullum'])

            train_loss, train_interaction = self.train_epoch(self.train_loaders[loader], WANDB, GREEDY_BASELINE, train_method, opts, config)
            if WANDB:
                wandb.log({"Avg Loss" : train_loss,
                            "epoch" : epoch + 1}, step = STEP)

            if self.distributed_context.is_leader:
                torch.cuda.empty_cache()
                metric = rand_neg_val(epoch + 1, WANDB, config = config)
            
                # Saving model
                if (SAVE_BEST_METRIC and metric > best_metric_score) or (opts.checkpoint_freq > 0 and (epoch + 1) % opts.checkpoint_freq==0): 
                    for idx, callback in enumerate(self.callbacks):
                        """
                        callbacks.ConsoleLogger: aggregated_print
                        finetuning.utils.ModelSaver: save_clipcap_model > {run_name}_e/final/best.pt                   
                        callbacks.CheckpointSaver: pass
                        """
                        callback.on_epoch_end(train_loss, train_interaction, epoch + 1, config['WANDB']['run_name'], SAVE_BEST_METRIC)
                        
                    if SAVE_BEST_METRIC:
                        best_metric_score = metric


    def load(self, checkpoint: Checkpoint):
        self.game.load_state_dict(checkpoint.model_state_dict, strict=False)
        self.optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        if checkpoint.optimizer_scheduler_state_dict:
            self.optimizer_scheduler.load_state_dict(
                checkpoint.optimizer_scheduler_state_dict
            )
        self.start_epoch = checkpoint.epoch

    def load_from_checkpoint(self, path):
        """
        Loads the game, agents, and optimizer state from a file
        :param path: Path to the file
        """
        print(f"# loading trainer state from {path}")
        checkpoint = torch.load(path)
        self.load(checkpoint)

    def load_from_latest(self, path):
        latest_file, latest_time = None, None

        for file in path.glob("*.tar"):
            creation_time = os.stat(file).st_ctime
            if latest_time is None or creation_time > latest_time:
                latest_file, latest_time = file, creation_time

        if latest_file is not None:
            self.load_from_checkpoint(latest_file)
    
    def save_val_preds(self, full_interaction, config, inference = False):
        preds = [j for i in full_interaction.message for j in i]
        cocoids = [i.item() for i in full_interaction.aux_input['cocoid']]
        val_preds =  dict(zip(cocoids, preds))

        if inference:
            save_path = os.path.join(config["inference"]["output_dir"], f"{config['captions_type']}_{config['opts']['checkpoint_dir'].split('/')[-1]}.pkl")                                        
            # save_path = os.path.join(config["inference"]["output_dir"], f"{config['captions_type']}_mle.pkl")                                        
        else:    
            save_path = os.path.join(config["opts"]["checkpoint_dir"].split("checkpoints")[0] + "val_preds", config["WANDB"]["run_name"] + f"_val_preds.pkl")                                        
        
        # print("$$$$"*100)
        # print(save_path)
        with open(save_path, "wb") as f:
            pickle.dump(val_preds, f)

# print(f" wte norm : {torch.norm(self.game.sender.clipcap.gpt.transformer.wte.weight)}")   
# print(f" clip_proj norm : {torch.norm(self.game.sender.clipcap.clip_project.model[0].weight)}")   
# print(f" wte grad : {torch.norm(self.game.sender.clipcap.gpt.transformer.wte.weight.grad)}")    

#check if clip param changing 
# print(f"mlp :  {torch.norm(self.game.sender.clip.visual.transformer.resblocks[0].mlp.c_proj.weight)}")
# print(f"attn :  {torch.norm(self.game.sender.clip.visual.transformer.resblocks[0].attn.out_proj.weight)}")

# check if A.B in lora changing
# dummy = self.game.sender.clipcap.gpt.transformer.h[0].attn.c_attn.parametrizations.weight
# dummy = self.game.sender.clip.visual.transformer.resblocks[0].attn.parametrizations
# dummy = self.game.sender.clip.visual.transformer.resblocks[0].mlp.c_proj.parametrizations.weight

# x = [(name, p) for name, p in dummy.named_parameters()]
# print("***"*30)
# print(f"grad A : {torch.norm(x[1][-1])}")
# print(f"grad B : {torch.norm(x[2][-1])}")
# print(f"A : {torch.norm(x[1][-1])}")
# print(f"B : {torch.norm(x[2][-1])}")
# print(f"lora (A.B).norm() : {torch.norm(torch.matmul(x[1][-1].t(),x[2][-1].t()))}")
# # print(f"og weights  :{torch.norm(x[0][-1])}")
# print("***"*30)
