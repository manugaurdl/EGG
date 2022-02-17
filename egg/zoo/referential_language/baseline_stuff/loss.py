import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.Module):
    def __init__(self, random_distractors: bool = False):
        super(Loss, self).__init__()
        self.random_distractors = random_distractors

    def forward(
        self,
        _sender_input,
        _message,
        _receiver_input,
        receiver_output,
        _labels,
        aux_input,
    ):
        labels = aux_input["game_labels"].view(-1)
        mask = aux_input["mask"].float().view(-1)

        bsz, max_objs = aux_input["mask"].shape
        all_accs = receiver_output.argmax(dim=-1) == labels
        aux_input["all_accs"] = all_accs.view(bsz, max_objs, -1).detach().float()

        if not self.training and self.random_distractors:
            acc_labels = torch.zeros(bsz, device=receiver_output.device)
            acc = (
                (receiver_output[0::max_objs].argmax(dim=-1) == acc_labels)
                .detach()
                .float()
            )
        else:
            acc = all_accs * mask  # zeroing masked elements
            acc = (acc.sum() / mask.sum()).unsqueeze(0)  # avoid dimensionless tensors

        loss = F.cross_entropy(receiver_output, labels, reduction="none")
        loss *= mask  # multiply by 0 masked elements

        return loss, {"acc": acc, "baseline": aux_input["baseline"]}
