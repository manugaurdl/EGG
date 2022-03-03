# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license fod in the
# LICENSE file in the root directory of this source tree.


import argparse
from collections import Counter, OrderedDict

import numpy as np
import torch

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


def get_msg_lists(msg, mask):
    n_batches, bsz, max_objs = mask.shape
    msg = torch.argmax(msg.view(n_batches, bsz, max_objs, 2, -1), dim=-1)

    nb_samples = 0
    list_sym1, list_sym2, list_msg = [], [], []
    for batch_id, batch in enumerate(msg):
        for elem_id, batch_elem in enumerate(batch):
            for msg_id, msg in enumerate(batch_elem):
                if mask[batch_id, elem_id, msg_id] is False:
                    continue
                nb_samples += 1
                list_sym1.append(msg[0].item())
                list_sym2.append(msg[1].item())
                list_msg.append(tuple(msg.tolist()))

    print(f" | Total number of samples is {nb_samples}")
    return list_sym1, list_sym2, list_msg


def analyze_msgs(msg, mask):
    stats = OrderedDict()
    l_sym1, l_sym2, l_msg = get_msg_lists(msg=msg, mask=mask)
    for idx, x in enumerate([l_sym1, l_sym2, l_msg]):
        stats[f"entropy_{idx}"] = calc_entropy(x)
        stats[f"maxent_{idx}"] = np.log2(x)
        stats[f"count_{idx}"] = len(Counter(x))
    return stats


def compute_accuracy(msgs, recv_input, receiver, loss, mask):
    acc = 0.0
    for batch_id in range(msgs.shape[0]):
        msg = msgs[batch_id]
        images = recv_input[batch_id]

        recv_output = receiver(msg, images)

        aux_input = {"mask": mask[batch_id]}
        _, aux = loss(None, None, None, recv_output, None, aux_input)
        acc += aux["acc"].item()

    print(f"acc = {acc / (batch_id + 1)}")


def intervene_msgs(l_sym1, l_sym2, l_sym3, msg):
    msg = msg.clone()
    for batch_id in range(msgs.shape[0]):
        msg = msgs[batch_id]
        for m in msg:
            idx = torch.argmax(m[0], -1)
            new_idx = counter_sym1.most_common(1)[0][0]
            if idx == new_idx:
                new_idx = counter_sym1.most_common(2)[1][0]
            # new_idx = random.sample(c1, k=1)[0]
            # while new_idx == idx:
            #    new_idx = random.sample(c1, k=1)[0]
            m[0][idx] = 0
            m[0][new_idx] = 1

    return msg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interaction_path")
    parser.add_argument("--checkpoint_path")
    parser.add_argument("--pdb", action="store_true", default=False)

    opts = parser.parse_args()
    if opts.pdb:
        global DEBUG
        DEBUG = True
        breakpoint()

    interaction = torch.load(opts.interaction_path)
    model = load_model(interaction.aux_input["args"], opts.checkpoint_path)

    analyze_msgs(msg=interaction.message, mask=interaction.aux_input["mask"])

    receiver = model.game.receiver
    loss = model.game.loss
    recv_input = interaction.aux_input["recv_img_feats"]
    msg = interaction.message
    mask = interaction.aux_input["mask"]
    compute_accuracy(msg, recv_input, receiver, loss, mask)


if __name__ == "__main__":
    main()
