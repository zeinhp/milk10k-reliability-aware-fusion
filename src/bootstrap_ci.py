"""
Phase 8: lesion-level bootstrap CI (B=2000, 95%) on macro F1, balanced accuracy, macro AUROC,
ECE, and AURC (entropy-based risk) -- per CLAUDE.md's "Statistical Analysis" section.

Reads the per-lesion predictions dumped by save_predictions.py (NOT evaluate.py's aggregate CSV --
bootstrap needs the raw per-lesion rows to resample). CPU-only, no GPU/checkpoints needed here.

For each (seed, modality_config), resamples lesions with replacement B times and reports the 95%
percentile CI. Also reports a seed-pooled summary per config (mean of the 5 per-seed point
estimates, plus the mean of their individual CIs) as a compact top-line table -- the per-seed CIs
remain the primary evidence, this is just for readability.

Usage (from src/):
    python bootstrap_ci.py --n_boot 2000

Output: ../outputs/results/bootstrap_ci_full_stage.csv (one row per seed x config)
        ../outputs/results/bootstrap_ci_full_stage_summary.md (pooled-per-config table)
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Bootstrap resamples of small/rare classes routinely produce a resample missing some class, or a
# resample with only one class present -- sklearn warns on every single such call (2000 resamples
# x 5 metrics x 35 cells = up to 350k calls). These warnings are expected and harmless here (the
# resulting metric value is still valid, e.g. zero_division=0 in macro_f1, or nan for AUROC which
# lesion_level_bootstrap_ci already drops via its try/except) -- printing all of them just floods
# the log and, more importantly, the repeated warnings.warn() calls were the actual bottleneck
# making this take hours instead of minutes. Suppress them; nothing here is silently wrong.
warnings.filterwarnings("ignore")

from metrics import (
    macro_f1, balanced_accuracy, macro_auroc, expected_calibration_error,
    predictive_entropy, risk_coverage_curve, aurc, lesion_level_bootstrap_ci,
)

NUM_CLASSES = 11
MODALITY_ORDER = ["C", "D", "M", "C,D", "C,M", "D,M", "C,D,M"]


def load_predictions(results_dir):
    files = sorted(results_dir.glob("predictions_full_stage*.csv"))
    assert files, f"No predictions_full_stage*.csv found in {results_dir} -- run save_predictions.py first."
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} prediction rows from {[f.name for f in files]}")
    return df


def probs_matrix(rows):
    return rows[[f"prob_{k}" for k in range(NUM_CLASSES)]].values


def metric_macro_f1(y_true, rows_or_probs):
    # bootstrap resample passes index-sliced arrays; we bootstrap over a combined array
    y_pred = rows_or_probs.argmax(axis=1)
    return macro_f1(y_true.astype(int), y_pred, NUM_CLASSES)


def metric_balanced_acc(y_true, probs):
    y_pred = probs.argmax(axis=1)
    return balanced_accuracy(y_true.astype(int), y_pred)


def metric_auroc(y_true, probs):
    return macro_auroc(y_true.astype(int), probs, NUM_CLASSES)


def metric_ece(y_true, probs):
    return expected_calibration_error(y_true.astype(int), probs)


def metric_aurc(y_true, probs):
    y_pred = probs.argmax(axis=1)
    correct = (y_pred == y_true.astype(int)).astype(float)
    entropy = predictive_entropy(probs)
    rng = entropy.max() - entropy.min()
    entropy_norm = (entropy - entropy.min()) / (rng if rng > 1e-8 else 1.0)
    cov, risk = risk_coverage_curve(entropy_norm, correct)
    return aurc(cov, risk)


METRICS = {
    "macro_f1": metric_macro_f1,
    "balanced_acc": metric_balanced_acc,
    "macro_auroc": metric_auroc,
    "ece": metric_ece,
    "aurc_entropy": metric_aurc,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--results_dir", default="../outputs/results")
    args = ap.parse_args()

    results_dir = Path(__file__).parent / args.results_dir
    df = load_predictions(results_dir)

    rows_out = []
    for seed in sorted(df["seed"].unique()):
        for cfg_key in MODALITY_ORDER:
            sub = df[(df["seed"] == seed) & (df["modality_config"] == cfg_key)]
            if len(sub) == 0:
                continue
            y_true = sub["label_true"].values.astype(int)
            probs = probs_matrix(sub)
            row = {"seed": seed, "modality_config": cfg_key, "n_lesions": len(sub)}
            for name, fn in METRICS.items():
                mean, lo, hi = lesion_level_bootstrap_ci(
                    fn, y_true, probs, n_boot=args.n_boot, ci=0.95, seed=0)
                row[f"{name}_mean"] = mean
                row[f"{name}_ci_lo"] = lo
                row[f"{name}_ci_hi"] = hi
            rows_out.append(row)
            print(f"seed{seed} {cfg_key}: macro_f1={row['macro_f1_mean']:.4f} "
                  f"[{row['macro_f1_ci_lo']:.4f}, {row['macro_f1_ci_hi']:.4f}]")

    out_df = pd.DataFrame(rows_out)
    out_csv = results_dir / "bootstrap_ci_full_stage.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")

    # Pooled-per-config summary: mean of the 5 seeds' point estimates and mean of their CI bounds.
    summary_lines = ["# Bootstrap CI Summary (Full stage, lesion-level, B={}, 95% CI)\n".format(args.n_boot),
                      "Per-config, averaged across the 5 seeds' individual bootstrap CIs "
                      "(each seed's CI is its own model's uncertainty; this average is for a "
                      "compact top-line view, not a re-bootstrap across seeds).\n",
                      "| Config | Macro F1 (95% CI) | Balanced Acc (95% CI) | AUROC (95% CI) | "
                      "ECE (95% CI) | AURC (95% CI) |",
                      "|---|---|---|---|---|---|"]
    for cfg_key in MODALITY_ORDER:
        sub = out_df[out_df["modality_config"] == cfg_key]
        if len(sub) == 0:
            continue

        def fmt(metric):
            m = sub[f"{metric}_mean"].mean()
            lo = sub[f"{metric}_ci_lo"].mean()
            hi = sub[f"{metric}_ci_hi"].mean()
            return f"{m:.4f} [{lo:.4f}, {hi:.4f}]"

        summary_lines.append(
            f"| {cfg_key} | {fmt('macro_f1')} | {fmt('balanced_acc')} | {fmt('macro_auroc')} | "
            f"{fmt('ece')} | {fmt('aurc_entropy')} |")

    out_md = results_dir / "bootstrap_ci_full_stage_summary.md"
    out_md.write_text("\n".join(summary_lines) + "\n")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
