# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Compute number of distinct names with contextual distractors compared to random distractors."""

import argparse
from collections import defaultdict, Counter

import numpy as np
import torch
from torch.nn.functional import cosine_similarity


def entropy_dict(freq_table):
    t = torch.tensor([v for v in freq_table.values()]).float()
    if (t < 0.0).any():
        raise RuntimeError("Encountered negative probabilities")

    t /= t.sum()
    return -(torch.where(t > 0, t.log(), t) * t).sum().item() / np.log(2)


def compute_errors(
    accs: torch.Tensor,
    labels: torch.Tensor,
    receiver_output: torch.Tensor,
    img_feats: torch.Tensor,
    attn_weights: torch.Tensor = None,
):
    cosine_sims = cosine_similarity(img_feats.unsqueeze(1), img_feats.unsqueeze(2), 3)
    max_objects = cosine_sims.shape[-1]
    self_mask = torch.eye(max_objects).fill_diagonal_(float("-inf"))
    cosine_sims += self_mask

    top_model_guesses = torch.argmax(receiver_output, dim=-1)
    top_visual_similarities = torch.argmax(cosine_sims, dim=-1)

    if attn_weights is not None:
        attn_weights = torch.argmax(attn_weights, dim=-1)
        visual_err_att_when_correct = 0
        visual_err_att_when_wrong = 0

    total_samples = 0
    total_err, label_err, visual_err, both_err = 0, 0, 0, 0
    for batch_id, acc in enumerate(accs):
        for obj_id, is_obj_guess_correct in enumerate(acc):
            if is_obj_guess_correct == -1:
                continue
            total_samples += 1

            idx_most_similar = top_visual_similarities[batch_id, obj_id]
            if is_obj_guess_correct == 0:
                total_err += 1
                idx = top_model_guesses[batch_id, obj_id].item()
                assert idx != obj_id

                chosen_label = labels[batch_id, idx]
                correct_label = labels[batch_id, obj_id]

                label = chosen_label == correct_label
                visual = idx == idx_most_similar

                label_err += 1 if label else 0
                visual_err += 1 if visual else 0

                both_err += 1 if visual and label else 0
                if attn_weights is not None:
                    if attn_weights[batch_id, obj_id] == idx_most_similar:
                        visual_err_att_when_wrong += 1
            else:
                if attn_weights is not None:
                    if attn_weights[batch_id, obj_id] == idx_most_similar:
                        visual_err_att_when_correct += 1

    if attn_weights is not None:
        print(
            f"Most similar distractor picked by  attn when correct = {visual_err_att_when_correct / total_samples:.2f}"
        )
        print(
            f"Most similar distractor picked by attn when wrong = {visual_err_att_when_wrong / total_samples:.2f}"
        )
    print(f"Visual errs: {visual_err / total_err * 100:.2f}%")
    print(f"Label errs: {label_err / total_err * 100:.2f}%")
    print(f"Both errs: {both_err / total_err * 100:.2f}%")


def compute_message_stats(
    labels: torch.Tensor,
    messages: torch.Tensor,
    accs: torch.Tensor,
):
    # defaultdict of default dict of int
    # from labels to messages to count of those messages
    labels2messages = defaultdict(lambda: defaultdict(int))
    all_message_counter = Counter()
    for batch_id, acc in enumerate(accs):
        for obj_id, acc_value in enumerate(acc):
            if acc_value == -1:
                continue
            message = messages[batch_id, obj_id].item()
            label = labels[batch_id, obj_id].int().item()
            all_message_counter[message] += 1
            labels2messages[label][message] += 1

    print(f"Number of distinct messages = {len(all_message_counter)}")
    messages_per_label = 0
    prop_entropy_per_label = 0
    for label, message_table in labels2messages.items():
        messages_per_label += len(message_table)
        if len(message_table) > 1:
            max_ent = np.log2(len(message_table))
            prop_entropy_per_label += entropy_dict(message_table) / max_ent

    nb_labels = len(labels2messages)
    print(f"Avg messages per label: {messages_per_label / nb_labels:.2f}")
    print(f"Avg proportional ent per label: {prop_entropy_per_label / nb_labels:.2f}")


def analyze_context(aux_input):
    if "context_gate" in aux_input:
        print("CONTEXT GATE")
        context_gate = aux_input["context_gate"]
        print(f"Mean = {torch.mean(context_gate):.2f}")
        print(f"Avg var is {torch.mean(torch.var(context_gate, dim=-1)):.4f}")
        print(f"Var of means {torch.var(torch.mean(context_gate, dim=-1)):.4f}")

    if "attn_weights" in aux_input:
        print("ATTN WEIGHTS")
        attn_weights = aux_input["attn_weights"]
        print(f"Var means attn weights: {torch.var(torch.mean(attn_weights, dim=-1))}")
        print(f"Mean attn weights: {torch.mean(attn_weights):.4f}")


def analyze_interaction(interaction):
    # print(interaction.aux_input["args"])

    messages = torch.argmax(interaction.message, -1)
    labels = interaction.labels.int()
    acc = interaction.aux_input["single_accs"]

    attn_wgt = None
    if "attn_weights" in interaction.aux_input:
        attn_wgt = interaction.aux_input["attn_weights"]
    compute_errors(
        acc,
        labels,
        interaction.receiver_output,
        interaction.aux_input["recv_img_feats"],
        attn_wgt,
    )
    analyze_context(interaction.aux_input)
    compute_message_stats(labels, messages, acc)

    print(f"Accuracy = {interaction.aux['acc'].mean().item()}")


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interaction_path")
    return parser.parse_args()


def main():
    opts = get_opts()
    interaction = torch.load(opts.interaction_path)
    analyze_interaction(interaction)


if __name__ == "__main__":
    main()
