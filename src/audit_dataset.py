"""
Phase 1 - Dataset audit. Run this BEFORE freezing any split.

Checks:
  - class distribution (11 classes)
  - missing metadata per field
  - per-class lesion counts vs. min_class_count_warning threshold (flags classes too rare to
    safely stratify into 70/10/20)
  - duplicate lesion_ids
  - image file existence (catches broken paths before training wastes time on them)
  - age/sex/site/skin_tone_class distributions

Usage (from src/):
    python audit_dataset.py
Output: prints a report to stdout and writes outputs/results/dataset_audit.json
"""
import json
import sys
from pathlib import Path

from common import load_config, resolve_data_paths, load_lesion_table


def main():
    config_dir = Path(__file__).parent / ".." / "configs"
    cfg = load_config(config_dir / "config.yaml")
    paths = resolve_data_paths(cfg, config_dir)

    for name, p in paths.items():
        if not Path(p).exists():
            print(f"ERROR: expected path does not exist: {name} -> {p}")
            print("Did you run extract_local.py and set data.data_root in config.yaml correctly?")
            sys.exit(1)

    lesions = load_lesion_table(cfg, paths)
    class_names = cfg["classes"]["names"]
    min_warn = cfg["split"]["min_class_count_warning"]

    report = {}

    # 1. class distribution
    class_counts = lesions["label_name"].value_counts().reindex(class_names, fill_value=0)
    report["class_counts"] = class_counts.to_dict()
    rare_classes = class_counts[class_counts < min_warn]
    report["rare_classes_below_threshold"] = rare_classes.to_dict()

    # 2. duplicate lesion ids
    dup = lesions["lesion_id"].duplicated().sum()
    report["duplicate_lesion_ids"] = int(dup)

    # 3. missing metadata
    missing = {}
    for col in ["age_approx", "sex", "site", "skin_tone_class"]:
        n_missing = lesions[col].isna().sum()
        missing[col] = {"missing_count": int(n_missing),
                         "missing_pct": round(100 * n_missing / len(lesions), 2)}
    report["missing_metadata"] = missing

    # 4. demographic distributions
    report["sex_distribution"] = lesions["sex"].value_counts(dropna=False).to_dict()
    report["site_distribution"] = lesions["site"].value_counts(dropna=False).to_dict()
    report["skin_tone_distribution"] = lesions["skin_tone_class"].value_counts(dropna=False).to_dict()
    report["age_stats"] = {
        "min": float(lesions["age_approx"].min()),
        "max": float(lesions["age_approx"].max()),
        "mean": round(float(lesions["age_approx"].mean()), 2),
        "missing": int(lesions["age_approx"].isna().sum()),
    }

    # 5. image existence check (sample-based to keep this fast; full check optional)
    sample = lesions.sample(min(500, len(lesions)), random_state=0)
    missing_files = []
    for _, row in sample.iterrows():
        if not Path(row["clinical_path"]).exists():
            missing_files.append(row["clinical_path"])
        if not Path(row["dermoscopy_path"]).exists():
            missing_files.append(row["dermoscopy_path"])
    report["sampled_missing_image_files"] = missing_files[:20]
    report["sampled_missing_image_files_count"] = len(missing_files)
    report["sample_size_checked"] = len(sample)

    # 6. total counts
    report["total_lesions"] = len(lesions)
    report["total_classes"] = len(class_names)

    out_dir = Path(__file__).parent / ".." / "outputs" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "dataset_audit.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    # human-readable summary
    print("=" * 70)
    print("MILK10k DATASET AUDIT")
    print("=" * 70)
    print(f"Total lesions: {report['total_lesions']}")
    print(f"Duplicate lesion_ids: {report['duplicate_lesion_ids']}")
    print("\nClass distribution:")
    for k, v in report["class_counts"].items():
        flag = "  <-- BELOW min_class_count_warning" if v < min_warn else ""
        print(f"  {k:10s} {v:5d}{flag}")
    if report["rare_classes_below_threshold"]:
        print(f"\n*** ACTION NEEDED: {len(report['rare_classes_below_threshold'])} class(es) "
              f"below threshold ({min_warn}). Decide explicitly (merge/exclude/accept) before "
              f"freezing the split — see CLAUDE.md split section. ***")
    print("\nMissing metadata:")
    for col, m in report["missing_metadata"].items():
        print(f"  {col:16s} missing={m['missing_count']:5d} ({m['missing_pct']}%)")
    print(f"\nSampled image existence check: {report['sampled_missing_image_files_count']} "
          f"missing out of {2 * report['sample_size_checked']} checked")
    print(f"\nFull report written to {out_dir / 'dataset_audit.json'}")


if __name__ == "__main__":
    main()
