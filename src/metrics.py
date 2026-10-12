"""
Evaluation metrics: Macro F1 (primary), balanced accuracy, macro AUROC, calibration (ECE, Brier,
NLL), predictive entropy/confidence, Jensen-Shannon disagreement, risk-coverage / selective
prediction, and lesion-level bootstrap CIs.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import (f1_score, balanced_accuracy_score, roc_auc_score,
                              brier_score_loss)


def macro_f1(y_true, y_pred, num_classes):
    return f1_score(y_true, y_pred, labels=list(range(num_classes)), average="macro", zero_division=0)


def balanced_accuracy(y_true, y_pred):
    return balanced_accuracy_score(y_true, y_pred)


def macro_auroc(y_true, y_probs, num_classes):
    y_true_onehot = np.eye(num_classes)[y_true]
    try:
        return roc_auc_score(y_true_onehot, y_probs, average="macro", multi_class="ovr")
    except ValueError:
        return float("nan")  # e.g. a class absent from y_true in a small eval slice


def expected_calibration_error(y_true, y_probs, n_bins=15):
    confidences = y_probs.max(axis=1)
    predictions = y_probs.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return ece


def multiclass_brier_score(y_true, y_probs, num_classes):
    y_true_onehot = np.eye(num_classes)[y_true]
    return np.mean(np.sum((y_probs - y_true_onehot) ** 2, axis=1))


def nll(y_true, y_probs, eps=1e-8):
    p_true = y_probs[np.arange(len(y_true)), y_true]
    return -np.mean(np.log(np.clip(p_true, eps, 1.0)))


def predictive_entropy(y_probs, eps=1e-8):
    p = np.clip(y_probs, eps, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def confidence(y_probs):
    return y_probs.max(axis=1)


def pairwise_js_divergence(p, q, eps=1e-8):
    """p, q: (N, K) probability arrays. Returns (N,) JS divergence per row, base-2 (bounded [0,1])."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    q = q / q.sum(axis=1, keepdims=True)
    return np.array([jensenshannon(p[i], q[i], base=2) ** 2 for i in range(len(p))])


def global_disagreement(p_c, p_d, p_m):
    """Mean pairwise JS divergence across the three per-modality prediction distributions.
    Only meaningful when all three modalities are available for a sample."""
    d_cd = pairwise_js_divergence(p_c, p_d)
    d_cm = pairwise_js_divergence(p_c, p_m)
    d_dm = pairwise_js_divergence(p_d, p_m)
    return (d_cd + d_cm + d_dm) / 3.0, {"D_CD": d_cd, "D_CM": d_cm, "D_DM": d_dm}


def risk_score(entropy, disagreement, lam):
    """R(x) = lambda * H(p) + (1 - lambda) * D_global. entropy and disagreement should be
    normalized to comparable scales by the caller before combining (e.g. min-max on validation)."""
    return lam * entropy + (1 - lam) * disagreement


def risk_coverage_curve(risk_scores, correct, n_points=100):
    """Returns (coverage[], selective_risk[]) sweeping the abstention threshold tau from
    "abstain on everything" to "abstain on nothing"."""
    order = np.argsort(risk_scores)  # lowest risk first -> predicted first as tau relaxes
    correct_sorted = correct[order]
    n = len(correct)
    coverages, risks = [], []
    for k in range(1, n + 1):
        coverages.append(k / n)
        risks.append(1 - correct_sorted[:k].mean())
    return np.array(coverages), np.array(risks)


def aurc(coverages, selective_risks):
    # np.trapz was removed in newer NumPy (renamed np.trapezoid in 2.0, and the deprecated
    # np.trapz alias is gone entirely in some installed versions here) -- avoid depending on
    # either name and just do the trapezoidal rule directly, which works on every NumPy version.
    coverages = np.asarray(coverages)
    selective_risks = np.asarray(selective_risks)
    return float(np.sum((selective_risks[1:] + selective_risks[:-1]) / 2.0 *
                         np.diff(coverages)))


def error_detection_auroc(risk_scores, correct):
    """AUROC for using risk_score to detect errors (label = 1 if incorrect)."""
    incorrect = 1 - correct
    if incorrect.sum() == 0 or incorrect.sum() == len(incorrect):
        return float("nan")
    return roc_auc_score(incorrect, risk_scores)


def lesion_level_bootstrap_ci(metric_fn, y_true, y_pred_or_probs, n_boot=2000, ci=0.95, seed=0,
                               **metric_kwargs):
    """
    Generic lesion-level bootstrap CI. metric_fn should accept (y_true_sample, y_pred_sample,
    **metric_kwargs) and return a scalar. Resamples lesion indices WITH replacement.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    values = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        try:
            v = metric_fn(y_true[idx], y_pred_or_probs[idx], **metric_kwargs)
        except Exception:
            continue
        values.append(v)
    values = np.array(values, dtype=float)
    # Some metrics (e.g. macro_auroc via roc_auc_score) return nan -- not raise -- when a
    # bootstrap resample happens to be missing a class entirely (common for rare classes with a
    # handful of test lesions). np.percentile/np.mean propagate nan if even one value is nan, which
    # silently turned every AUROC bootstrap CI into nan regardless of how many of the 2000
    # resamples were actually fine. Use the nan-aware variants so a resample that hit a missing
    # class is just dropped from the estimate, same as an exception would be.
    finite = values[~np.isnan(values)]
    if len(finite) == 0:
        return float("nan"), float("nan"), float("nan")
    lo = np.nanpercentile(values, (1 - ci) / 2 * 100)
    hi = np.nanpercentile(values, (1 + ci) / 2 * 100)
    return float(np.nanmean(values)), float(lo), float(hi)


def paired_bootstrap_delta(metric_fn, y_true, pred_a, pred_b, n_boot=2000, ci=0.95, seed=0,
                            **metric_kwargs):
    """95% CI for metric(model_a) - metric(model_b) using the SAME bootstrap resample for both
    models each iteration (paired), as required for a valid paired comparison."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    deltas = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        try:
            va = metric_fn(y_true[idx], pred_a[idx], **metric_kwargs)
            vb = metric_fn(y_true[idx], pred_b[idx], **metric_kwargs)
        except Exception:
            continue
        deltas.append(va - vb)
    deltas = np.array(deltas)
    lo = np.percentile(deltas, (1 - ci) / 2 * 100)
    hi = np.percentile(deltas, (1 + ci) / 2 * 100)
    return float(np.mean(deltas)), float(lo), float(hi)
