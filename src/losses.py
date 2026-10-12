"""Loss functions, including the reliability-aware model's auxiliary correctness loss."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def classification_loss(logits, labels):
    return F.cross_entropy(logits, labels)


def reliability_auxiliary_loss(aux_logits: dict, reliability: dict, labels, availability):
    """
    L_reliability = mean over available modalities of BCE(r_m, correct_m)
    where correct_m = 1[argmax(aux_logits[m]) == label], computed with aux_logits detached
    (correctness is a supervision target, not something we backprop classification gradient
    through here — the aux head itself is trained via aux_classification_loss below).
    availability: (B, 3) tensor in [C, D, M] order, used only to know which items in the batch
    actually had modality m turned on (aux_logits/reliability dicts only contain keys for
    modalities present in at least one item of the active-modality set for this forward pass;
    within a fixed-modality batch this is uniform, but this function is written to be safe even
    if used with per-item varying availability in a custom training loop).
    """
    modality_idx = {"C": 0, "D": 1, "M": 2}
    losses = []
    for m, logits_m in aux_logits.items():
        with torch.no_grad():
            pred_m = logits_m.argmax(dim=-1)
            correct_m = (pred_m == labels).float()
        mask = availability[:, modality_idx[m]]
        if mask.sum() < 1:
            continue
        bce = F.binary_cross_entropy(reliability[m], correct_m, reduction="none")
        losses.append((bce * mask).sum() / mask.sum().clamp_min(1))
    if not losses:
        return torch.tensor(0.0, device=labels.device)
    return torch.stack(losses).mean()


def aux_classification_loss(aux_logits: dict, labels):
    """Trains each per-modality auxiliary head to actually predict the label — this is what
    makes correct_m in reliability_auxiliary_loss meaningful, not degenerate."""
    losses = [F.cross_entropy(logits_m, labels) for logits_m in aux_logits.values()]
    if not losses:
        return torch.tensor(0.0, device=labels.device)
    return torch.stack(losses).mean()


def reliability_aware_total_loss(outputs, labels, availability, lambda_r=0.5, lambda_aux=1.0):
    """
    Combines: main classification loss (fusion head) + aux per-modality classification loss
    (so aux heads are meaningful) + reliability BCE loss (so R_m has a real supervision signal).
    Returns (total_loss, components_dict) so train.py can log each term separately.
    """
    l_cls = classification_loss(outputs["logits"], labels)
    l_aux = aux_classification_loss(outputs["aux_logits"], labels)
    l_rel = reliability_auxiliary_loss(outputs["aux_logits"], outputs["reliability"], labels,
                                        availability)
    total = l_cls + lambda_aux * l_aux + lambda_r * l_rel
    return total, {"cls": l_cls.item(), "aux": l_aux.item(), "reliability": l_rel.item()}
