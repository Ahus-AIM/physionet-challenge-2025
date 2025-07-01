from typing import List, Tuple

import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def fix_input(
    logits_list: List[torch.Tensor] | torch.Tensor, labels_list: List[torch.Tensor] | torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    if (type(logits_list) is list) and (type(labels_list) is list):
        logits = torch.cat(logits_list, dim=0).squeeze().float()
        labels = torch.cat(labels_list, dim=0).squeeze().float()
    elif (type(logits_list) is torch.Tensor) and (type(labels_list) is torch.Tensor):
        logits = logits_list.squeeze().float()
        labels = labels_list.squeeze().float()
    else:
        raise ValueError("logits and labels must be either list of tensors or tensors")

    if labels.dim() != 1 or logits.dim() != 1:
        raise ValueError("labels and logits must be 1D tensors")
    if labels.shape != logits.shape:
        raise ValueError("labels and logits must have the same shape")
    return logits, labels


def challenge_score(
    logits_list: List[torch.Tensor] | torch.Tensor,
    labels_list: List[torch.Tensor] | torch.Tensor,
    max_fraction_positive: float = 0.05,
) -> float:

    logits, labels = fix_input(logits_list, labels_list)

    num_instances = labels.size(0)
    max_num_positive_instances = int(max_fraction_positive * num_instances)

    sorted_logits, indices = torch.sort(logits, descending=True)
    sorted_labels = labels[indices]

    tp = torch.cumsum(sorted_labels, dim=0)
    fp = torch.cumsum(1 - sorted_labels, dim=0)
    fn = tp[-1] - tp

    threshold_idx = torch.searchsorted(tp + fp, max_num_positive_instances, right=True)
    tpr = (tp[threshold_idx] / (tp[threshold_idx] + fn[threshold_idx])).item() if tp[-1] > 0 else float("nan")

    return float(tpr)


def aucroc(logits_list: List[torch.Tensor] | torch.Tensor, labels_list: List[torch.Tensor] | torch.Tensor) -> float:
    logits, labels = fix_input(logits_list, labels_list)
    roc: float = float(roc_auc_score(labels.numpy(), logits.numpy()))
    return roc


def aucprc(logits_list: List[torch.Tensor] | torch.Tensor, labels_list: List[torch.Tensor] | torch.Tensor) -> float:
    logits, labels = fix_input(logits_list, labels_list)
    prc: float = float(average_precision_score(labels.numpy(), logits.numpy()))
    return prc


if __name__ == "__main__":
    labels = torch.zeros(10000)
    labels[:99] = 1
    logits = torch.randn(10000) ** 2
    logits[:1000] += 10

    print(challenge_score(logits, labels))
    from helper_code import compute_challenge_score

    probs_list = torch.sigmoid(logits).tolist()
    labels_list = labels.tolist()
    print(compute_challenge_score(labels_list, probs_list))
