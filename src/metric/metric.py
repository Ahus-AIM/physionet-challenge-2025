from typing import List, Tuple, Union

import torch
from sklearn.metrics import average_precision_score, roc_auc_score

TensorOrList = Union[torch.Tensor, List[torch.Tensor]]


def fix_input(logits_list: TensorOrList, labels_list: TensorOrList) -> Tuple[torch.Tensor, torch.Tensor]:
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
    labels = (labels > 0.5).float()
    return logits, labels


def _flatten_logits_labels(
    logits_list: TensorOrList, labels_list: TensorOrList, ignore_index: int = -100
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    - logits_list: either a single tensor of shape (B, C, T) or a list of such tensors
    - labels_list: either a single tensor of shape (B, T) or a list of such tensors
    Returns:
      logits_flat: (N, C) tensor
      labels_flat: (N,) tensor
    where N = number of valid (non-ignore) entries across all batches/tests.
    """
    # concat if needed
    if isinstance(logits_list, list):
        logits = torch.cat(logits_list, dim=0)
    else:
        logits = logits_list
    if isinstance(labels_list, list):
        labels = torch.cat(labels_list, dim=0)
    else:
        labels = labels_list

    # sanity
    if logits.dim() != 3:
        raise ValueError(f"logits must be 3D (B,C,T), got {logits.shape}")
    if labels.dim() != 2:
        raise ValueError(f"labels must be 2D (B,T), got {labels.shape}")
    if logits.shape[0] != labels.shape[0] or logits.shape[2] != labels.shape[1]:
        raise ValueError("Batch and test dimensions of logits and labels must match")

    B, C, T = logits.shape
    # Mask out ignore_index
    mask = labels != ignore_index  # shape (B,T)
    # Flatten
    logits_flat = logits.permute(0, 2, 1)[mask]  # (N, C)
    labels_flat = labels[mask]  # (N,)
    return logits_flat, labels_flat


def topk_accuracy(
    logits_list: TensorOrList,
    labels_list: TensorOrList,
    k: int = 10,
    ignore_index: int = -100,
) -> float:
    """
    Compute Top-k accuracy over all non-ignored (B,T) positions.

    Args:
      logits_list: tensor or list of (B, C, T)
      labels_list: tensor or list of (B, T)
      k: top-k
      ignore_index: label value to ignore

    Returns:
      top-k accuracy in [0,1] as float
    """
    logits_flat, labels_flat = _flatten_logits_labels(logits_list, labels_list, ignore_index)
    if labels_flat.numel() == 0:
        return float("nan")  # no valid entries

    # get top-k predictions per instance
    # topk_inds: (N, k)
    topk_vals, topk_inds = logits_flat.topk(k, dim=1, largest=True, sorted=True)

    # compare true label against each row's top-k
    # labels_flat.unsqueeze(1): (N,1) -> compare to (N,k)
    correct = (topk_inds == labels_flat.unsqueeze(1)).any(dim=1).float()
    return correct.mean().item()


def challenge_score(
    logits_list: TensorOrList,
    labels_list: TensorOrList,
    alpha: float = 0.05,
) -> float:
    logits, labels = fix_input(logits_list, labels_list)
    k = int(alpha * len(labels))
    topk_indices = torch.topk(logits, k).indices
    return labels[topk_indices].sum().item() / labels.sum().clamp(min=1).item()


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
