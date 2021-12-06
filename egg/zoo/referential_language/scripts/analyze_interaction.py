# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from collections import defaultdict
from typing import List

import torch
from torch.nn.functional import cosine_similarity


def get_correctly_shaped_acc(
    acc: torch.Tensor, mask: torch.Tensor, max_objs: int, n_batches: int, bsz: int
) -> List[List[torch.Tensor]]:
    """Transform a 1-D tensor of flattened binary accuracies into a list of list of Tensors.

    This in order to account for the variably sized tensor given by the variable number of objects in each image.
    We are basically removing padding/masking by turning the (flatten) tensor into a list of list of tensors.
    Each element is 1 if that object was correctly discriminated, else 0.

    The first list is of len = n_batches and contains lists each of len = batch_size.
    Each of this nested list has a tensor of variable size depending on the number of objects but alwyas <= max_objs
    """
    acc_list = []
    start_index = 0
    for batch_id, batch_mask in enumerate(mask):
        acc_list.append([])
        for mask_elem in batch_mask:
            end_index = start_index + max_objs - int(mask_elem.item())
            acc_list[batch_id].append(acc[start_index:end_index])
            start_index = end_index
    assert len(acc_list) == n_batches
    assert [len(elem) for elem in acc_list] == [bsz] * n_batches
    return acc_list


def print_errors(
    labels: torch.Tensor,
    receiver_output: torch.Tensor,
    acc_list: List[List[torch.Tensor]],
    recv_img_feats: torch.Tensor,
):
    same_class_error, wrong_class_error = 0, 0
    total_errors, total_guesses = 0, 0
    visual_errors = 0
    both_errors = 0

    cosine_sim = cosine_similarity(
        recv_img_feats.unsqueeze(2), recv_img_feats.unsqueeze(3), 4
    )
    # a.masked_select(~torch.eye(n, dtype=bool)).view(n, n - 1)

    # Each tensor in acc_list  has the same number of elements as the number of objects in each img max (<= max_objects)
    #   and each element is 1 if that object was correctly discriminated, else 0.
    for batch_id, batch in enumerate(acc_list):
        for img_id, img_accs in enumerate(batch):
            # dim: max_objects X max_objects
            model_guesses = receiver_output[batch_id, img_id]
            total_guesses += img_accs.shape[0]

            # dim: max_objects
            # we get only the first choice of the receiver for each object
            sorted_idx_accs = torch.argsort(model_guesses, descending=True)[:, 0]

            sorted_sims = torch.argsort(cosine_sim[batch_id, img_id], descending=True)

            wrong_guesses = torch.nonzero((img_accs == 0)).squeeze(-1).tolist()
            for guess_id in wrong_guesses:
                total_errors += 1

                right_guess_label = labels[batch_id, img_id, guess_id]
                wrong_guess_label = labels[batch_id, img_id, sorted_idx_accs[guess_id]]

                if sorted_idx_accs[guess_id] == sorted_sims[guess_id, 1]:
                    visual_errors += 1
                if right_guess_label == wrong_guess_label:
                    same_class_error += 1
                else:
                    wrong_class_error += 1

                if (
                    sorted_idx_accs[guess_id] == sorted_sims[guess_id, 1]
                    and right_guess_label == wrong_guess_label
                ):
                    both_errors += 1

    assert same_class_error + wrong_class_error == total_errors
    print(f"Total errors: {total_errors}")
    print(f"Visual_errors: {visual_errors / total_errors * 100:.2f}%")
    print(f"Same error perc: {same_class_error / total_errors * 100:.2f}%")
    print(
        f"Both errors (visually similar and same class): {both_errors / total_errors * 100:.2f}%"
    )
    print(f"Errors: {total_errors / total_guesses * 100:.2f}%")


def get_distinct_message_counter(
    labels: torch.Tensor, messages: torch.Tensor, acc_list: List[List[torch.Tensor]]
):
    # defaultdict of default dict of int
    # from labels to messages to count of those messages
    labels2message = defaultdict(lambda: defaultdict(int))
    for batch_id, batch_accs in enumerate(acc_list):
        for img_id, accs in enumerate(batch_accs):
            for elem_id, elem in enumerate(accs):
                label = labels[batch_id, img_id, elem_id].int().item()
                message = torch.argmax(messages[batch_id, img_id, elem_id]).item()
                labels2message[label][message] += 1
    return labels2message


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interaction_path", help="Run the game with pdb enabled")
    return parser.parse_args()


def main():
    # Compute number of distinct names with contextual distractors compared to random distractors
    opts = get_opts()
    interaction = torch.load(opts.interaction_path)

    # Extracting standard-ish fields from the interaction
    run_args = interaction.aux_input["args"]
    print(run_args)
    bsz, max_objs = run_args["batch_size"], run_args["max_objects"]
    labels = interaction.labels.view(-1, bsz, max_objs)
    receiver_output = interaction.receiver_output.view(-1, bsz, max_objs, max_objs)
    n_batches = receiver_output.shape[0]
    messages = interaction.message.view(n_batches, bsz, max_objs, -1)
    acc = interaction.aux["acc"]
    total_errors = sum(acc == 0)
    total_guesses = len(acc)  # it's a flatten vector of accuracies

    # Extracting values from aux_input
    aux_input = interaction.aux_input
    _ = aux_input["img_ids"].view(-1, bsz)
    _ = aux_input["obj_ids"].view(-1, bsz, max_objs)
    # We assume original images are always at 224^2 resolution
    _ = aux_input["original_imgs"].view(-1, bsz, 3, 224, 224)
    recv_img_feats = aux_input["recv_img_feats"].view(n_batches, bsz, max_objs, -1)
    mask = aux_input["mask"].view(n_batches, bsz)

    acc_list = get_correctly_shaped_acc(acc, mask, max_objs, n_batches, bsz)

    print_errors(labels, receiver_output, acc_list, recv_img_feats)
    labels2message = get_distinct_message_counter(labels, messages, acc_list)
    messages_per_class = 0
    for k, v in labels2message.items():
        messages_per_class += len(v)
    print(f"Average messages per class: {messages_per_class / len(labels2message):.2f}")

    err_perc = total_errors / total_guesses
    print(f"Accuracy: {(1 - err_perc) * 100:.2f}%")


if __name__ == "__main__":
    main()
