# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Compute number of distinct names with contextual distractors compared to random distractors."""

import argparse
from collections import defaultdict, Counter
from typing import Optional

import numpy as np
import torch
from torch.nn.functional import cosine_similarity


def entropy_dict(freq_table):
    t = torch.tensor([v for v in freq_table.values()]).float()
    if (t < 0.0).any():
        raise RuntimeError("Encountered negative probabilities")

    t /= t.sum()
    return -(torch.where(t > 0, t.log(), t) * t).sum().item() / np.log(2)


# a.masked_select(~torch.eye(n, dtype=bool)).view(n, n - 1)
def get_errors(
    accs: torch.Tensor,
    labels: torch.Tensor,
    receiver_output: torch.Tensor,
    img_feats: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
):
    cosine_sims = cosine_similarity(img_feats.unsqueeze(2), img_feats.unsqueeze(3), 4)
    top_visual_similarities = torch.argsort(cosine_sims, descending=True)[:, :, :, 1]
    top_model_guesses = torch.argsort(receiver_output, descending=True)[:, :, :, 0]

    total_errors = 0
    label_errors, visual_errors = 0, 0
    potential_label_errors = 0
    both_errors = 0
    for batch_id, _ in enumerate(accs):
        for batch_elem_id, _ in enumerate(accs[batch_id]):
            counter = Counter(labels[batch_id, batch_elem_id].tolist())
            for obj_id, model_guess in enumerate(accs[batch_id, batch_elem_id]):
                wrong_guess = model_guess.item() == 0
                not_masked = mask[batch_id, batch_elem_id, obj_id].item() == 0
                if not_masked and wrong_guess:
                    total_errors += 1
                    idx = top_model_guesses[batch_id, batch_elem_id, obj_id]
                    chosen_label = labels[batch_id, batch_elem_id, idx]
                    correct_label = labels[batch_id, batch_elem_id, obj_id]

                    if counter[correct_label.item()] > 1:
                        potential_label_errors += 1

                    label_err = chosen_label == correct_label
                    label_errors += 1 if label_err else 0
                    visual_err = (
                        idx == top_visual_similarities[batch_id, batch_elem_id, obj_id]
                    )
                    visual_errors += 1 if visual_err else 0
                    both_errors += 1 if visual_err and label_err else 0

    return (
        visual_errors,
        label_errors,
        both_errors,
        potential_label_errors,
        total_errors,
    )


def get_message_info(
    labels: torch.Tensor,
    messages: torch.Tensor,
    accs: torch.Tensor,
    mask: torch.Tensor,
):
    # defaultdict of default dict of int
    # from labels to messages to count of those messages
    labels2message = defaultdict(lambda: defaultdict(int))
    all_messages = defaultdict(bool)
    for batch_id, _ in enumerate(accs):
        for batch_elem_id, _ in enumerate(accs[batch_id]):
            for obj_id, model_guess in enumerate(accs[batch_id, batch_elem_id]):
                not_masked = mask[batch_id, batch_elem_id, obj_id].item() == 0
                if not_masked:
                    label = labels[batch_id, batch_elem_id, obj_id].int().item()
                    message = messages[batch_id, batch_elem_id, obj_id]
                    message = tuple(message.tolist())
                    if message not in all_messages:
                        all_messages[messages] = True
                    labels2message[label][message] += 1
    return labels2message, all_messages


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interaction_path", help="Run the game with pdb enabled")
    return parser.parse_args()


def analyze_interaction(interaction):
    # Extracting standard-ish fields from the interaction
    run_args = interaction.aux_input["args"]
    print(run_args)
    bsz, max_objs = run_args.batch_size, run_args.max_objects
    msg_len = run_args.max_len + 1 if run_args.max_len > 1 else run_args.max_len

    receiver_output = interaction.receiver_output.view(-1, bsz, max_objs, max_objs)
    n_batches = receiver_output.shape[0]
    messages = interaction.message.view(n_batches, bsz, max_objs, msg_len, -1).squeeze()
    messages = torch.argmax(messages, -1)
    labels = interaction.labels.view(-1, bsz, max_objs)
    acc = interaction.aux["acc"].view(-1, bsz, max_objs)

    # Extracting values from aux_input
    aux_input = interaction.aux_input
    recv_img_feats = aux_input["recv_img_feats"].view(n_batches, bsz, max_objs, -1)
    mask = aux_input["mask"].view(n_batches, bsz, max_objs)

    (
        visual_errors,
        label_errors,
        both_errors,
        potential_label_errors,
        total_errors,
    ) = get_errors(acc, labels, receiver_output, recv_img_feats, mask)
    print(f"Visual errs: {visual_errors / total_errors * 100:.2f}%")
    print(f"Label errs: {label_errors / total_errors * 100:.2f}%")
    print(f"Potential Label errs: {potential_label_errors / total_errors * 100:.2f}%")
    print(f"Both errs: {both_errors / total_errors * 100:.2f}%")

    labels2messages, all_messages = get_message_info(labels, messages, acc, mask)
    print(f"Number of distinct messages = {len(all_messages)}")
    messages_per_label = 0
    prop_entropy_per_label = 0
    labels_with_syn = 0
    for label, message_table in labels2messages.items():
        messages_per_label += len(message_table)
        if len(message_table) > 1:
            max_ent = np.log2(len(message_table))
            prop_entropy_per_label += entropy_dict(message_table) / max_ent
            labels_with_syn += 1

    nb_labels = len(labels2messages)
    print(f"Avg messages per label: {messages_per_label / nb_labels:.2f}")
    print(
        f"Avg proportional ent per label: {prop_entropy_per_label / labels_with_syn:.2f}"
    )

    print(f"Accuracy = {torch.sum(acc == 1).int() / acc.numel() * 100:.2f}%")


def main():
    opts = get_opts()
    interaction = torch.load(opts.interaction_path)
    analyze_interaction(interaction)


if __name__ == "__main__":
    main()
