"""
Phase 8: subgroup performance and calibration analysis (per CLAUDE.md -- explicitly NOT a
"fairness" claim, just a breakdown). Subgroups fixed BEFORE looking at results, per protocol:
  - sex: male / female (as recorded; other values reported as-is, not collapsed)
  - age band: <40, 40-59, >=60 (age_approx)
  - skin_tone_class: MILK10k's own 0-5 scale

Reads the per-lesion predictions dumped by save_predictions.py. CPU-only. Reported on the
C+D+M (full modality) config only, Full ablation stage -- the primary/complete model, since
subgroup breakdown for all 7 configs x 3 subgroup axes would be a very large, hard-to-read table
and the primary model is what a subgroup claim should be about.

Usage (from src/):
    python subgroup_analysis.py

Output: ../outputs/results/subgroup_analysis_full_stage.csv (one row per seed x subgroup axis x group)
        ../outputs/results/subgroup_analysis_full_stage_summary.md
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import macro_f1, expected_calibration_error

NUM_CLASSES = 11


def age_band(age):
    if pd.isna(age):
        return "unknown"
    if age < 40:
        return "<40"
    if age < 60:
        return "40-59"
    return ">=60"


def sensitivity_macro(y_true, y_pred, num_classes):
    """Macro-averaged recall (= sensitivity), same averaging convention as macro F1."""
    sens = []
    for c in range(num_classes):
        mask = y_true == c
        if mask.sum() == 0:
            continue
        sens.append((y_pred[mask] == c).mean())
    return float(np.mean(sens)) if sens else float("nan")


def compute_group_metrics(sub):
    y_true = sub["label_true"].values.astype(int)
    y_pred = sub["label_pred"].values.astype(int)
    probs = sub[[f"prob_{k}" for k in range(NUM_CLASSES)]].values
    error_rate = float((y_pred != y_true).mean())
    return {
        "n": len(sub),
        "macro_f1": macro_f1(y_true, y_pred, NUM_CLASSES),
        "sensitivity_macro": sensitivity_macro(y_true, y_pred, NUM_CLASSES),
        "ece": expected_calibration_error(y_true, probs) if len(sub) > 0 else float("nan"),
        "error_rate": error_rate,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="../outputs/results")
    args = ap.parse_args()

    results_dir = Path(__file__).parent / args.results_dir
    files = sorted(results_dir.glob("predictions_full_stage*.csv"))
    assert files, f"No predictions_full_stage*.csv found in {results_dir} -- run save_predictions.py first."
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df["modality_config"] == "C,D,M"].copy()
    df["age_band"] = df["age_approx"].apply(age_band)
    print(f"{len(df)} full-modality (C,D,M) rows across {df['seed'].nunique()} seeds.")

    axes = {"sex": "sex", "age_band": "age_band", "skin_tone_class": "skin_tone_class"}
    rows_out = []
    for axis_name, col in axes.items():
        for seed in sorted(df["seed"].unique()):
            seed_df = df[df["seed"] == seed]
            for group_val in sorted(seed_df[col].dropna().unique().tolist(), key=str):
                sub = seed_df[seed_df[col] == group_val]
                if len(sub) < 5:
                    continue  # too few lesions for a meaningful metric
                m = compute_group_metrics(sub)
                rows_out.append({"seed": seed, "axis": axis_name, "group": group_val, **m})

    out_df = pd.DataFrame(rows_out)
    out_csv = results_dir / "subgroup_analysis_full_stage.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    lines = ["# Subgroup Performance and Calibration Analysis (Full stage, C+D+M config)\n",
             "Averaged across the 5 seeds. Groups with <5 lesions in a given seed's test split "
             "are excluded from that seed (too few to compute a meaningful per-group metric); "
             "the n below is the average n across seeds where the group had enough lesions.\n"]
    for axis_name in axes:
        axis_df = out_df[out_df["axis"] == axis_name]
        if len(axis_df) == 0:
            continue
        lines.append(f"## {axis_name}\n")
        lines.append("| Group | n (avg) | Macro F1 | Sensitivity (macro) | ECE | Error rate |")
        lines.append("|---|---|---|---|---|---|")
        for group_val, g in axis_df.groupby("group"):
            lines.append(f"| {group_val} | {g['n'].mean():.0f} | {g['macro_f1'].mean():.4f} | "
                          f"{g['sensitivity_macro'].mean():.4f} | {g['ece'].mean():.4f} | "
                          f"{g['error_rate'].mean():.4f} |")
        lines.append("")

    out_md = results_dir / "subgroup_analysis_full_stage_summary.md"
    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
