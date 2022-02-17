import torch
import torch.nn as nn


class RandomDistractorsTestLoss(nn.Module):
    def forward(
        self,
        _sender_input,
        _message,
        _receiver_input,
        receiver_output,
        _labels,
        aux_input,
    ):
        bsz, max_objs = aux_input["mask"].shape

        labels = aux_input["game_labels"].view(-1)
        all_accs = (receiver_output.argmax(dim=-1) == labels).detach().float()
        aux_input["all_accs"] = all_accs.view(bsz, max_objs, -1)

        labels = torch.zeros(bsz, device=receiver_output.device)
        acc = (receiver_output[0::max_objs].argmax(dim=-1) == labels).detach().float()
        return torch.zeros(1), {"acc": acc}
