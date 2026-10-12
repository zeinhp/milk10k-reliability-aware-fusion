"""
Phase 6-7 evaluation (CLAUDE.md): 7-config modality sweep, calibration, disagreement, and
uncertainty / selective prediction for the reliability_aware model.

Loads each reliability_aware checkpoint (one per seed x ablation_stage, 30 total) ONCE, then runs
inference under all 7 mandatory modality configs (C, D, M, C+D, C+M, D+M, C+D+M) against that
seed's held-out TEST split (never used for training or checkpoint selection -- val was already
used for early stopping / best-epoch selection, so evaluating on val again here would be
optimistically biased).

For each (seed, ablation_stage, modality_config) cell, computes:
  - macro F1, balanced accuracy, macro AUROC  (primary classification metrics)
  - ECE, Brier score, NLL                      (calibration)
  - mean predictive entropy, mean confidence   (uncertainty)
  - mean pairwise Jensen-Shannon disagreement across the ACTIVE per-modality prediction
    distributions (nan for single-modality configs, where disagreement is undefined)
  - AURC (area under risk-coverage curve) and error-detection AUROC, twice: once using entropy
    alone as the risk score (defined for every config), and once using entropy+disagreement
    combined (only defined for multi-modality configs) -- this is the core Phase 7 "selective
    prediction" analysis: how well does the model's own uncertainty signal predict its errors?

This is READ-ONLY inference over existing checkpoints -- does not train or modify anything.
Resume-safe: results are appended to a CSV row-by-row as they're computed, and any
(seed, stage, config) combination already present in that CSV is skipped on a re-run, so an
interrupted sweep can just be restarted with the same command.

Usage (from src/):
    python evaluate.py --device cuda:0
    python evaluate.py --device cuda:0 --seeds 0,1 --stages A1,Full   # partial sweep, e.g. to test

Output: ../outputs/results/evaluate_reliability_aware_sweep.csv (one row per seed x stage x config)
"""
import argparse
import csv
import itertools
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import load_config, resolve_data_paths, load_lesion_table
from dataset import MILK10kDataset, build_categorical_vocabs
from models import Encoders, build_model
from metrics import (
    macro_f1, balanced_accuracy, macro_auroc, expected_calibration_error,
    multiclass_brier_score, nll, predictive_entropy, confidence, pairwise_js_divergence,
    risk_score, risk_coverage_curve, aurc, error_detection_auroc,
)

SEEDS_DEFAULT = [0, 1, 2, 3, 4]
STAGES_DEFAULT = ["A1", "A2", "A3", "A4", "A5", "Full"]

CSV_FIELDS = [
    "seed", "ablation_stage", "modality_config", "n_samples",
    "macro_f1", "balanced_acc", "macro_auroc",
    "ece", "brier", "nll",
    "mean_entropy", "mean_confidence", "mean_disagreement",
    "aurc_entropy", "error_auroc_entropy",
    "aurc_combined", "error_auroc_combined",
]


def parse_list_arg(s, cast=str):
    return [cast(x) for x in s.split(",")] if s else None


def load_existing_keys(csv_path):
    """Returns the set of (seed, stage, config) tuples already present in the results CSV, so a
    re-run can skip them -- makes the sweep resumable after an interruption."""
    if not csv_path.exists():
        return set()
    keys = set()
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            keys.add((int(row["seed"]), row["ablation_stage"], row["modality_config"]))
    return keys


def append_row(csv_path, row):
    is_new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if is_new:
            w.writeheader()
        w.writerow(row)


def run_inference(model, loader, device):
    """Returns y_true, y_pred, y_probs (fused prediction) and per_modality_probs_list (one dict
    per batch, m -> (B, K) probs, only for modalities active in that batch's fixed config)."""
    all_labels, all_preds, all_probs = [], [], []
    per_modality_batches = []
    with torch.no_grad():
        for batch in loader:
            for k in ["clinical_img", "dermoscopy_img", "availability", "label"]:
                batch[k] = batch[k].to(device)
            for k in batch["metadata"]:
                batch["metadata"][k] = batch["metadata"][k].to(device)
            active = tuple(m for i, m in enumerate(["C", "D", "M"]) if batch["availability"][0, i] > 0)
            out = model(batch, active_letters=active)
            probs = torch.softmax(out["logits"], dim=-1).cpu().numpy()
            all_probs.append(probs)
            all_preds.append(probs.argmax(axis=1))
            all_labels.append(batch["label"].cpu().numpy())
            per_modality_batches.append(
                {m: p.detach().cpu().numpy() for m, p in out["per_modality_probs"].items()})
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_probs = np.concatenate(all_probs)
    return y_true, y_pred, y_probs, per_modality_batches


def mean_disagreement_for_config(per_modality_batches, active_modalities):
    """Average pairwise JS divergence across all pairs of active modalities, batched. Undefined
    (nan) for single-modality configs."""
    if len(active_modalities) < 2:
        return None
    pair_means = []
    for m1, m2 in itertools.combinations(sorted(active_modalities), 2):
        batch_js = []
        for pmb in per_modality_batches:
            batch_js.append(pairwise_js_divergence(pmb[m1], pmb[m2]))
        pair_means.append(np.concatenate(batch_js))
    stacked = np.stack(pair_means, axis=0)  # (n_pairs, N)
    return stacked.mean(axis=0)  # (N,) per-sample mean disagreement across pairs


def evaluate_one(model, loader, device, num_classes, active_modalities):
    y_true, y_pred, y_probs, per_modality_batches = run_inference(model, loader, device)
    n = len(y_true)
    correct = (y_pred == y_true).astype(float)

    entropy = predictive_entropy(y_probs)
    conf = confidence(y_probs)
    disagreement = mean_disagreement_for_config(per_modality_batches, active_modalities)

    row = {
        "n_samples": n,
        "macro_f1": macro_f1(y_true, y_pred, num_classes),
        "balanced_acc": balanced_accuracy(y_true, y_pred),
        "macro_auroc": macro_auroc(y_true, y_probs, num_classes),
        "ece": expected_calibration_error(y_true, y_probs),
        "brier": multiclass_brier_score(y_true, y_probs, num_classes),
        "nll": nll(y_true, y_probs),
        "mean_entropy": float(entropy.mean()),
        "mean_confidence": float(conf.mean()),
        "mean_disagreement": float(disagreement.mean()) if disagreement is not None else "",
    }

    # Selective prediction: does the model's own uncertainty predict its errors? Entropy-only risk
    # is defined for every config; entropy+disagreement combined risk only for multi-modality ones.
    entropy_norm = (entropy - entropy.min()) / max(entropy.max() - entropy.min(), 1e-8)
    cov_e, risk_e = risk_coverage_curve(entropy_norm, correct)
    row["aurc_entropy"] = aurc(cov_e, risk_e)
    row["error_auroc_entropy"] = error_detection_auroc(entropy_norm, correct)

    if disagreement is not None:
        dis_norm = (disagreement - disagreement.min()) / max(disagreement.max() - disagreement.min(), 1e-8)
        combined = risk_score(entropy_norm, dis_norm, lam=0.5)
        cov_c, risk_c = risk_coverage_curve(combined, correct)
        row["aurc_combined"] = aurc(cov_c, risk_c)
        row["error_auroc_combined"] = error_detection_auroc(combined, correct)
    else:
        row["aurc_combined"] = ""
        row["error_auroc_combined"] = ""

    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--config", default="../configs/config.yaml",
                     help="any of the project's configs work here (model architecture fields are "
                          "identical across config.yaml / config_gpu1080_reliability.yaml / "
                          "config_gpu1080_baseline.yaml -- only training.batch_size differs, and "
                          "that has no effect on evaluation).")
    ap.add_argument("--seeds", default=None, help="comma-separated, default all of 0,1,2,3,4")
    ap.add_argument("--stages", default=None, help="comma-separated, default all of A1..Full")
    ap.add_argument("--split", default="test", choices=["val", "test"],
                     help="which split to evaluate on. Default test (held-out, never used for "
                          "training or checkpoint selection). Only use val for a quick sanity "
                          "check -- val was already used to pick each checkpoint's best epoch, so "
                          "reporting val numbers here would be optimistically biased.")
    ap.add_argument("--eval_batch_size", type=int, default=32,
                     help="independent of training batch_size -- no gradients here, so this can "
                          "be larger than what training used without any VRAM concern.")
    ap.add_argument("--csv_suffix", default="",
                     help="appended to the output CSV filename, e.g. '_gpu0' -> "
                          "evaluate_reliability_aware_sweep_gpu0.csv. Use DIFFERENT suffixes when "
                          "running this sweep on two GPUs at once (e.g. each doing a different "
                          "subset of --seeds) so both processes never append to the exact same "
                          "file concurrently -- concurrent appends to one CSV from two processes "
                          "can interleave mid-row and corrupt it. Merge the separate CSVs "
                          "afterward (they're disjoint by seed, so a plain concatenation is fine).")
    args = ap.parse_args()

    seeds = parse_list_arg(args.seeds, int) or SEEDS_DEFAULT
    stages = parse_list_arg(args.stages, str) or STAGES_DEFAULT

    config_dir = Path(__file__).parent / Path(args.config).parent
    cfg = load_config(Path(__file__).parent / args.config)
    paths = resolve_data_paths(cfg, config_dir)
    num_classes = len(cfg["classes"]["names"])
    modality_configs = [set(mc) for mc in cfg["modality_configs"]]  # 7 mandatory configs

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    lesions_all = load_lesion_table(cfg, paths)
    vocabs = build_categorical_vocabs(lesions_all, cfg["metadata_fields"]["categorical"])

    ckpt_dir = Path(__file__).parent / cfg["output"]["checkpoints_dir"]
    results_dir = Path(__file__).parent / cfg["output"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"evaluate_reliability_aware_sweep{args.csv_suffix}.csv"
    done = load_existing_keys(csv_path)
    print(f"{len(done)} (seed, stage, config) cells already in {csv_path.name}, will be skipped.")

    splits_dir = Path(__file__).parent / cfg["split"]["out_dir"]
    total_cells = len(seeds) * len(stages) * len(modality_configs)
    done_count = 0

    for seed in seeds:
        split_csv = splits_dir / f"seed{seed}_{args.split}.csv"
        assert split_csv.exists(), f"Missing split file {split_csv} -- run make_splits.py first."

        # Build one loader per modality config, reused across every ablation stage for this seed
        # (the dataset/loader only depends on seed + config, not on which checkpoint is loaded).
        loaders = {}
        for mc in modality_configs:
            key = ",".join(sorted(mc))
            ds = MILK10kDataset(split_csv, cfg, args.split, vocabs, fixed_modalities=mc)
            loaders[key] = DataLoader(ds, batch_size=args.eval_batch_size, shuffle=False,
                                       num_workers=0)

        for stage in stages:
            ckpt_path = ckpt_dir / f"reliability_aware_CDM_seed{seed}_{stage}_best.pt"
            model = None  # loaded lazily below, only if there's at least one cell left to do

            for mc in modality_configs:
                config_key = ",".join(sorted(mc))
                cache_key = (seed, stage, config_key)
                done_count += 1
                if cache_key in done:
                    continue
                if not ckpt_path.exists():
                    print(f"SKIP seed{seed}/{stage}: checkpoint not found at {ckpt_path}")
                    break

                if model is None:
                    encoders = Encoders(vocabs, cfg).to(device)
                    model = build_model("reliability_aware", encoders, num_classes, cfg).to(device)
                    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                    model.load_state_dict(ckpt["model_state"])
                    model.eval()

                print(f"[{done_count}/{total_cells}] seed{seed} {stage} config={config_key} ...",
                      end=" ", flush=True)
                row = evaluate_one(model, loaders[config_key], device, num_classes, mc)
                row.update({"seed": seed, "ablation_stage": stage, "modality_config": config_key})
                append_row(csv_path, row)
                print(f"macro_f1={row['macro_f1']:.4f}")

    print(f"\nDone. Results in {csv_path}")


if __name__ == "__main__":
    main()
