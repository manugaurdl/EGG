# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license fod in the
# LICENSE file in the root directory of this source tree.


import argparse
import json
import random
from collections import Counter, OrderedDict
from random import sample
from typing import List

import numpy as np
import torch
import torch.nn as nn
from timebudget import timebudget

from egg.core.language_analysis import calc_entropy
from egg.zoo.referential_language.games import build_game

DEBUG = False


def load_model(model_args, checkpoint_path):
    model = build_game(model_args)
    model_ckpt = torch.load(checkpoint_path)
    model.load_state_dict(model_ckpt.model_state_dict)
    if DEBUG:
        print(f"| Successfully loaded model from {checkpoint_path}")
    return model


def get_msg_lists(msgs, mask, stats=None):
    n_batches, bsz, max_objs = mask.shape
    msgs = torch.argmax(msgs.view(n_batches, bsz, max_objs, 2, -1), dim=-1)

    nb_samples = 0
    list_sym1, list_sym2, list_msg = [], [], []
    for batch_id, batch in enumerate(msgs):
        for elem_id, batch_elem in enumerate(batch):
            for msg_id, msg in enumerate(batch_elem):
                if mask[batch_id, elem_id, msg_id].item() is False:
                    continue
                nb_samples += 1
                list_sym1.append(msg[0].item())
                list_sym2.append(msg[1].item())
                list_msg.append(tuple(msg.tolist()))

    if stats is not None:
        stats["nb_samples"] = nb_samples

    return list_sym1, list_sym2, list_msg


def compute_msg_stats(interaction, stats):
    msgs = interaction.message
    mask = interaction.aux_input["mask"]

    l_sym1, l_sym2, l_msg = get_msg_lists(msgs=msgs, mask=mask, stats=stats)
    counters = [Counter(x) for x in [l_sym1, l_sym2, l_msg]]
    for idx, c in enumerate(counters):
        stats[f"entropy_{idx + 1}"] = round(calc_entropy(c), 4)
        stats[f"maxent_{idx + 1}"] = round(np.log2(len(c)), 4)
        stats[f"count_{idx + 1}"] = len(c)
    stats["msg_intersection"] = len(counters[0] & counters[1])
    return stats


def compute_accuracy(msgs, recv_input, receiver, loss, mask, stats=None):
    acc = 0.0
    receiver.eval()
    with torch.no_grad():
        for batch_id in range(msgs.shape[0]):
            msg = msgs[batch_id]
            images = recv_input[batch_id]

            recv_output = receiver(msg, images)

            aux_input = {"mask": mask[batch_id]}
            _, aux = loss(None, None, None, recv_output, None, aux_input)
            acc += aux["acc"].item()

    acc /= batch_id + 1
    acc = round(acc, 4)

    if stats is None:
        return OrderedDict({"acc": acc})

    return acc


def edit_msgs(
    symb_list: List[int], msgs: torch.Tensor, sym: int, fn: str, mask: torch.Tensor
):
    assert sym in [0, 1, 2]
    assert fn in ["random", "most_common", "swap"]
    n_batches, bsz, max_objs = mask.shape
    msgs = msgs.view(n_batches, bsz, max_objs, 2, -1)

    new_msgs = msgs.clone()

    counter = Counter(symb_list)

    def idxs_are_equal(idx, new_idx):
        if isinstance(new_idx, tuple):
            new_idx = list(new_idx)
        return idx == new_idx

    for batch_id, batch in enumerate(new_msgs):
        for elem_id, batch_elem in enumerate(batch):
            for msg_id, msg in enumerate(batch_elem):
                if mask[batch_id, elem_id, msg_id] is False:
                    continue

                if sym == 2:
                    idx = list(torch.argmax(msg, -1))
                else:
                    idx = torch.argmax(msg[sym], -1).item()

                if fn == "random":
                    new_idx = sample(symb_list, k=1)[0]
                    while idxs_are_equal(idx, new_idx):
                        new_idx = sample(symb_list, k=1)[0]
                elif fn == "most_common":
                    new_idx = counter.most_common(1)[0][0]
                    if idxs_are_equal(idx, new_idx):
                        new_idx = counter.most_common(2)[1][0]
                elif fn == "swap":
                    new_idx = list(torch.argmax(msg, -1))[::-1]

                if isinstance(idx, int):
                    msg[sym][idx] = 0
                    msg[sym][new_idx] = 1
                else:
                    for i, (idx, new_idx) in enumerate(zip(idx, new_idx)):
                        msg[i][idx] = 0
                        msg[i][new_idx] = 1

    return new_msgs.view(n_batches, bsz * max_objs, 2, -1)


def intervention_msgs(interaction, receiver, loss, stats):
    recv_input = interaction.aux_input["recv_img_feats"]
    msgs = interaction.message
    mask = interaction.aux_input["mask"]

    l_sym1, l_sym2, l_msg = get_msg_lists(msgs=msgs, mask=mask)
    for idx, l in enumerate([l_sym1, l_sym2, l_msg]):
        for fn in ["random", "most_common", "swap"]:
            if idx != 2 and fn == "swap":
                continue
            k = f"intervention_sym{idx}_{fn}"
            new_msg = edit_msgs(l, msgs, sym=idx, fn=fn, mask=mask)
            stats[k] = compute_accuracy(
                new_msg, recv_input, receiver, loss, mask, stats=stats
            )
    return stats


def compute_attn_stats(interaction, stats):
    attn_weights = interaction.aux_input["attn_weights"]
    n_batches, bsz, max_objs, max_objs = attn_weights.squeeze().shape
    max_values = torch.max(attn_weights, -1)[0].view(-1)
    stats["avg_max_attn"] = round(torch.mean(max_values).item(), 4)
    stats["std_max_attn"] = round(torch.std(max_values).item(), 4)
    return stats


def intervention_target(game, interaction, stats):
    mask = interaction.aux_input["mask"]
    attn_weights = torch.argmax(interaction.aux_input["attn_weights"].squeeze(), -1)
    sender_img_feats = interaction.aux_input["sender_img_feats"]
    recv_img_feats = interaction.aux_input["recv_img_feats"]
    labels = interaction.labels

    n_batches, bsz, max_objs = mask.shape
    num_changes, total_samples = 0, 0
    acc = 0.0

    game.eval()
    with torch.no_grad():
        for batch_id in range(n_batches):
            sender_input = sender_img_feats[batch_id]
            recv_input = recv_img_feats[batch_id]
            new_aux_input = {"mask": mask[batch_id]}
            batch_attn_weights = attn_weights[batch_id]

            for elem_id in range(bsz):
                n_elem_id = random.sample(range(bsz), k=1)
                n_obj_id = random.sample(range(max_objs), k=1)

                while (
                    mask[batch_id, n_elem_id, n_obj_id] is False or n_elem_id == elem_id
                ):
                    n_elem_id = random.sample(bsz, k=1)
                    n_obj_id = random.sample(max_objs, k=1)

                s_inp = sender_input.clone()
                s_inp[elem_id][0] = sender_input[n_elem_id, n_obj_id]

                r_inp = recv_input.clone()
                r_inp[elem_id][0] = recv_input[n_elem_id, n_obj_id]

                _, n_interaction = game(s_inp, labels, r_inp, new_aux_input)

                old_pick = batch_attn_weights[elem_id][0]
                new_pick = torch.argmax(
                    n_interaction.aux_input["attn_weights"].squeeze()[elem_id][0], -1
                )
                if new_pick != old_pick:
                    num_changes += 1

                acc += n_interaction.aux_input["guesses"][elem_id][0].squeeze().item()

                total_samples += 1

    stats["ratio_attn_changes"] = round(num_changes / total_samples, 4)
    stats["avg_acc_attn_changes"] = round(acc / total_samples, 4)
    stats["total_attn_changes"] = total_samples
    return stats


class HardcodeAttnSender(nn.Module):
    def __init__(self, msg_generator):
        super(HardcodeAttnSender, self).__init__()
        self.msg_generator = msg_generator

    def forward(self, x, aux_input=None):
        attn = aux_input["hardcoded_attn"]
        return self.msg_generator(x, attn, aux_input)


def intervention_context(game, interaction, stats):
    n_batches, bsz, max_objs = interaction.aux_input["mask"].shape
    acc, msg_changes, total_samples = 0.0, 0, 0

    def _sample_distractor():
        new_batch_id = random.sample(range(n_batches), k=1)
        new_elem_id = random.sample(range(bsz), k=1)
        new_obj_id = random.sample(range(max_objs), k=1)
        while (
            interaction.aux_input["mask"][new_batch_id, new_elem_id, new_obj_id]
            is False
        ):
            new_batch_id = (batch_id + 1) % n_batches
            new_elem_id = random.sample(range(bsz), k=1)
            new_obj_id = random.sample(range(max_objs), k=1)
        img_feats = interaction.aux_input["sender_img_feats"]
        return img_feats[new_batch_id, new_elem_id, new_obj_id]

    game.sender = HardcodeAttnSender(game.sender.msg_generator)
    game.eval()
    with torch.no_grad():
        for batch_id in range(n_batches):
            sender_input = interaction.aux_input["sender_img_feats"][batch_id]
            recv_input = interaction.aux_input["recv_img_feats"][batch_id]
            labels = interaction.labels[batch_id]
            mask = interaction.aux_input["mask"][batch_id]

            hardcoded_attn = [_sample_distractor() for _ in range(bsz * max_objs)]
            hardcoded_attn = torch.cat(hardcoded_attn)
            new_aux_input = {"mask": mask, "hardcoded_attn": hardcoded_attn}

            _, n_interaction = game(sender_input, labels, recv_input, new_aux_input)

            # only consider context-conditioned message
            n_message = torch.argmax(n_interaction.message, -1)[:, 1]
            old_message = torch.argmax(interaction.message[batch_id], -1)[:, 1]
            msg_mask = mask.view(-1)

            msg_changes += torch.sum((old_message != n_message).int() * msg_mask).item()
            acc += n_interaction.aux["acc"].item()

    total_samples = interaction.aux_input["mask"].int().sum().item()
    stats["ratio_msg_changes"] = round(msg_changes / total_samples, 4)
    stats["avg_acc_distractors_changes"] = round(acc / total_samples, 4)
    stats["total_distractor_changes"] = total_samples
    return stats


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interaction_path")
    parser.add_argument("--checkpoint_path")
    parser.add_argument("--pdb", action="store_true", default=False)

    opts = parser.parse_args()
    if opts.pdb:
        global DEBUG
        DEBUG = True
        breakpoint()

    seed = 111
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return opts


def perform_analysis(opts, interaction, model):
    # bypassing vision encoder as we already have visual feats in the interaction
    comm_game = model.game
    recv_input = interaction.aux_input["recv_img_feats"]
    msgs = interaction.message
    mask = interaction.aux_input["mask"]

    receiver = model.game.receiver
    loss = model.game.loss

    with timebudget("Compute first accuracy"):
        # compute initial accuracy values
        stats = compute_accuracy(msgs, recv_input, receiver, loss, mask)

    with timebudget("Message intervention analysis"):
        # analyze msg stats and run msg intervention analysis
        stats = compute_msg_stats(interaction, stats=stats)
        stats = intervention_msgs(interaction, receiver, loss, stats)

    #
    with timebudget("Target intervention analysis"):
        # compute stats on attn_weights and run target intervention analysis
        stats = compute_attn_stats(interaction, stats)
        stats = intervention_target(comm_game, interaction, stats=stats)

    with timebudget("Attended distractor intervention analysis"):
        # run distractrors(context)-weighted intervention analysis
        stats = intervention_context(comm_game, interaction, stats=stats)

    print(json.dumps(stats), flush=True)


def main():
    opts = get_opts()

    interaction = torch.load(opts.interaction_path)
    model = load_model(interaction.aux_input["args"], opts.checkpoint_path)

    perform_analysis(opts, interaction, model)


if __name__ == "__main__":
    main()
