"""
Phase 4 evaluation (CLAUDE.md): held-out TEST-split evaluation for the conventional baselines --
B1 unimodal (15 checkpoints: 3 modalities x 5 seeds) and B2/B3/B4 baseline_multimodal
(naive_concat / late_fusion / gated_fusion, 60 checkpoints: 3 families x 4 multi-modality configs
x 5 seeds).

Unlike evaluate.py (which sweeps all 7 modality configs against EACH reliability_aware checkpoint,
because that model is trained with modality dropout and supports every config from one checkpoint),
these baseline families are each trained on exactly ONE fixed --modality config (see train.py's
docstring and CLAUDE.md Section 19) -- so there is nothing to sweep: every checkpoint is evaluated
at exactly the single modality_config it was trained on. Together, the 4 multi-modality checkpoints
per family ARE the missing-modality robustness comparison point for that family (vs. one
reliability_aware checkpoint handling all 7 configs).

Metrics: macro F1, balanced accuracy, macro AUROC (primary), ECE / Brier / NLL (calibration),
mean predictive entropy / confidence (uncertainty). Disagreement / selective-prediction metrics
(AURC, error-detection AUROC) are NOT computed here -- unimodal/naive_concat/gated_fusion have no
per-modality branch outputs to disagree with each other (only late_fusion happens to expose
per_modality_probs, and only when >1 modality is active), so computing disagreement for 3 out of 4
families would make the comparison table apples-to-oranges. That analysis is reserved for the
proposed reliability_aware model (Phase 6-7, evaluate.py), which is the point of comparison.

This is READ-ONLY inference over existing checkpoints -- does not train or modify anything.
Resume-safe: results are appended to a CSV row-by-row as they're computed, and any
(family, modality_config, seed) combination already present in that CSV is skipped on a re-run.

Usage (from src/):
    python evaluate_baselines.py --device cuda:0

Output: ../outputs/results/evaluate_baselines_sweep.csv
        (one row per family x modality_config x seed -- 75 rows total: 15 unimodal + 60 multi)
"""
import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from common import load_config, resolve_data_paths, load_lesion_table
from dataset import MILK10kDataset, build_categorical_vocabs
from models import Encoders, build_model
from metrics import (
    macro_f1, balanced_accuracy, macro_auroc, expected_calibration_error,
    multiclass_brier_score, nll, predictive_entropy, confidence,
)

SEEDS_DEFAULT = [0, 1, 2, 3, 4]

# (family, modality_config) cells to evaluate. Unimodal gets the 3 single-letter configs;
# the 3 multi-modality baseline families each get the 4 mandatory multi-modality configs
# (singles are covered by unimodal -- see baseline_multimodal_commands() in run_all.py, which
# skips single-modality configs for these families for exactly this reason).
UNIMODAL_CONFIGS = ["C", "D", "M"]
MULTI_FAMILIES = ["naive_concat", "late_fusion", "gated_fusion"]
MULTI_CONFIGS = ["C,D", "C,M", "D,M", "C,D,M"]

CSV_FIELDS = [
    "family", "modality_config", "seed", "n_samples",
    "macro_f1", "balanced_acc", "macro_auroc",
    "ece", "brier", "nll",
    "mean_entropy", "mean_confidence",
]


def load_existing_keys(csv_path):
    if not csv_path.exists():
        return set()
    keys = set()
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            keys.add((row["family"], row["modality_config"], int(row["seed"])))
    return keys


def append_row(csv_path, row):
    is_new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if is_new:
            w.writeheader()
        w.writerow(row)


def run_inference(model, loader, device, family):
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            for k in ["clinical_img", "dermoscopy_img", "availability", "label"]:
                batch[k] = batch[k].to(device)
            for k in batch["metadata"]:
                batch["metadata"][k] = batch["metadata"][k].to(device)
            if family == "unimodal":
                out = model(batch)  # UnimodalClassifier.forward takes no active_letters arg
            else:
                active = tuple(m for i, m in enumerate(["C", "D", "M"]) if batch["availability"][0, i] > 0)
                out = model(batch, active_letters=active)
            probs = torch.softmax(out["logits"], dim=-1).cpu().numpy()
            all_probs.append(probs)
            all_preds.append(probs.argmax(axis=1))
            all_labels.append(batch["label"].cpu().numpy())
    import numpy as np
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_probs = np.concatenate(all_probs)
    return y_true, y_pred, y_probs


def evaluate_one(model, loader, device, num_classes, family):
    y_true, y_pred, y_probs = run_inference(model, loader, device, family)
    entropy = predictive_entropy(y_probs)
    conf = confidence(y_probs)
    return {
        "n_samples": len(y_true),
        "macro_f1": macro_f1(y_true, y_pred, num_classes),
        "balanced_acc": balanced_accuracy(y_true, y_pred),
        "macro_auroc": macro_auroc(y_true, y_probs, num_classes),
        "ece": expected_calibration_error(y_true, y_probs),
        "brier": multiclass_brier_score(y_true, y_probs, num_classes),
        "nll": nll(y_true, y_probs),
        "mean_entropy": float(entropy.mean()),
        "mean_confidence": float(conf.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--config", default="../configs/config.yaml",
                     help="architecture fields are identical across all project configs; only "
                          "training.batch_size differs, which has no effect on evaluation.")
    ap.add_argument("--seeds", default=None, help="comma-separated, default all of 0,1,2,3,4")
    ap.add_argument("--split", default="test", choices=["val", "test"],
                     help="default test (held-out). Only use val for a quick sanity check -- val "
                          "was already used to pick each checkpoint's best epoch.")
    ap.add_argument("--eval_batch_size", type=int, default=32)
    ap.add_argument("--csv_suffix", default="")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else SEEDS_DEFAULT

    config_dir = Path(__file__).parent / Path(args.config).parent
    cfg = load_config(Path(__file__).parent / args.config)
    paths = resolve_data_paths(cfg, config_dir)
    num_classes = len(cfg["classes"]["names"])

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
    csv_path = results_dir / f"evaluate_baselines_sweep{args.csv_suffix}.csv"
    done = load_existing_keys(csv_path)
    print(f"{len(done)} (family, modality_config, seed) cells already in {csv_path.name}, will be skipped.")

    splits_dir = Path(__file__).parent / cfg["split"]["out_dir"]

    # Build the full cell list: (family, modality_config, ckpt_name) for every seed.
    cells = []
    for mc in UNIMODAL_CONFIGS:
        cells.append(("unimodal", mc))
    for family in MULTI_FAMILIES:
        for mc in MULTI_CONFIGS:
            cells.append((family, mc))

    total = len(cells) * len(seeds)
    done_count = 0

    for seed in seeds:
        split_csv = splits_dir / f"seed{seed}_{args.split}.csv"
        assert split_csv.exists(), f"Missing split file {split_csv} -- run make_splits.py first."

        # One loader per distinct modality_config, reused across every family that shares it.
        loaders = {}
        for mc in UNIMODAL_CONFIGS + MULTI_CONFIGS:
            mc_set = set(mc.split(","))
            key = mc
            ds = MILK10kDataset(split_csv, cfg, args.split, vocabs, fixed_modalities=mc_set)
            loaders[key] = DataLoader(ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=0)

        for family, mc in cells:
            done_count += 1
            cache_key = (family, mc, seed)
            if cache_key in done:
                continue

            modality_no_comma = mc.replace(",", "")
            ckpt_name = f"{family}_{modality_no_comma}_seed{seed}_best.pt"
            ckpt_path = ckpt_dir / ckpt_name
            if not ckpt_path.exists():
                print(f"[{done_count}/{total}] SKIP {family}/{mc}/seed{seed}: checkpoint not found ({ckpt_name})")
                continue

            encoders = Encoders(vocabs, cfg).to(device)
            build_modality = mc if family == "unimodal" else None
            model = build_model(family, encoders, num_classes, cfg, modality=build_modality).to(device)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            model.eval()

            row = evaluate_one(model, loaders[mc], device, num_classes, family)
            row["family"] = family
            row["modality_config"] = mc
            row["seed"] = seed
            append_row(csv_path, row)
            done.add(cache_key)
            print(f"[{done_count}/{total}] {family}/{mc}/seed{seed}: "
                  f"macro_f1={row['macro_f1']:.4f} ece={row['ece']:.4f}", flush=True)

            del model, encoders
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print(f"Done. {csv_path} has {len(done)} cells.")


if __name__ == "__main__":
    main()
