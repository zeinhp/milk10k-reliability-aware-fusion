"""
Phase 2 - Freeze the lesion-level stratified 70/10/20 split, for 5 seeds.

Run AFTER audit_dataset.py has been reviewed and any rare-class decision has been made explicitly
(if audit flagged rare classes, resolve that first — this script does not silently merge classes).

Usage (from src/):
    python make_splits.py
Writes: ../splits/seed{N}_{train,val,test}.csv for N in config seeds, each with columns
[lesion_id, label, label_name, clinical_path, dermoscopy_path, age_approx, sex, site,
skin_tone_class].
Also writes ../splits/split_summary.json with per-seed per-split class counts, so imbalance is
visible/auditable, not hidden.
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from common import load_config, resolve_data_paths, load_lesion_table


def stratified_split(df, train_frac, val_frac, test_frac, seed, label_col="label"):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    train_df, temp_df = train_test_split(
        df, test_size=(val_frac + test_frac), stratify=df[label_col], random_state=seed)
    relative_test_frac = test_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(
        temp_df, test_size=relative_test_frac, stratify=temp_df[label_col], random_state=seed)
    return train_df, val_df, test_df


def main():
    config_dir = Path(__file__).parent / ".." / "configs"
    cfg = load_config(config_dir / "config.yaml")
    paths = resolve_data_paths(cfg, config_dir)
    lesions = load_lesion_table(cfg, paths)

    out_cols = ["lesion_id", "label", "label_name", "clinical_path", "dermoscopy_path",
                "age_approx", "sex", "site", "skin_tone_class"]

    # NOTE: cfg["split"]["out_dir"] (e.g. "../splits") is written relative to this src/ folder,
    # matching how train.py resolves it (splits_dir = Path(__file__).parent / out_dir) --
    # previously this had an extra "/.." here, writing one directory too high
    # (project_root/../splits instead of project_root/splits) so train.py could never find them.
    out_dir = Path(__file__).parent / cfg["split"]["out_dir"]
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for seed in cfg["split"]["seeds"]:
        train_df, val_df, test_df = stratified_split(
            lesions, cfg["split"]["train_frac"], cfg["split"]["val_frac"],
            cfg["split"]["test_frac"], seed)

        train_df[out_cols].to_csv(out_dir / f"seed{seed}_train.csv", index=False)
        val_df[out_cols].to_csv(out_dir / f"seed{seed}_val.csv", index=False)
        test_df[out_cols].to_csv(out_dir / f"seed{seed}_test.csv", index=False)

        summary[f"seed{seed}"] = {
            "train": {"n": len(train_df), "class_counts": train_df["label_name"].value_counts().to_dict()},
            "val": {"n": len(val_df), "class_counts": val_df["label_name"].value_counts().to_dict()},
            "test": {"n": len(test_df), "class_counts": test_df["label_name"].value_counts().to_dict()},
        }
        print(f"seed {seed}: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    with open(out_dir / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nSplits written to {out_dir}")
    print("IMPORTANT: treat seed0_test.csv .. seed4_test.csv as frozen from this point on. "
          "Do not regenerate splits after starting model development (that invalidates the "
          "held-out guarantee) unless the audit forced a re-run before any training happened.")


if __name__ == "__main__":
    main()
