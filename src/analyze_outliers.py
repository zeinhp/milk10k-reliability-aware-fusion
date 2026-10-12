"""
Ad-hoc error analysis for the two reliability_aware/1080-overflow runs flagged as possible
outliers in the 30-run recap (18 Aug): seed2/A5 (val macro F1 0.4427) and seed3/Full (0.4332).

Loads each flagged run's *_best.pt checkpoint, re-runs inference on that seed's own val split
(full C,D,M modality, matching training-time evaluation), and reports:
  - per-class F1 / support (which classes is it actually failing on?)
  - confusion matrix (11x11)
  - calibration (ECE) and mean predictive entropy/confidence
For comparison, does the same for a "normal" run at the same ablation stage (seed0), so the
per-class breakdown of the outlier can be read side by side with a run that scored well.

This does NOT retrain anything and does NOT touch any in-progress run -- pure read-only inference
over existing checkpoints, safe to run any time, does not conflict with A4000 or anything else.

Usage (from src/):
    python analyze_outliers.py

Output: printed to stdout AND written to ../outputs/results/outlier_analysis.md
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix

from common import load_config, resolve_data_paths, load_lesion_table
from dataset import MILK10kDataset, build_categorical_vocabs
from models import Encoders, build_model
from metrics import macro_f1, expected_calibration_error, predictive_entropy, confidence

# (seed, ablation_stage, label) -- the two flagged outliers plus a same-stage "normal" reference
# for comparison. All are family=reliability_aware, modality=C,D,M (only fixed combo trained).
RUNS_TO_ANALYZE = [
    (2, "A5", "seed2_A5 (OUTLIER, F1=0.4427)"),
    (0, "A5", "seed0_A5 (reference, F1=0.5957)"),
    (3, "Full", "seed3_Full (OUTLIER, F1=0.4332)"),
    (0, "Full", "seed0_Full (reference, F1=0.5549)"),
]

CONFIG_PATH = "../configs/config_gpu1080_reliability.yaml"  # matches how these were trained


def load_model_for_run(seed, stage, cfg, paths, vocabs, device):
    # cfg["output"]["checkpoints_dir"] (e.g. "../outputs/checkpoints") is already relative to
    # src/, same convention train.py uses -- resolve it the same way (Path(__file__).parent / ...)
    # rather than assuming the script's cwd is src/.
    ckpt_dir = Path(__file__).parent / cfg["output"]["checkpoints_dir"]
    ckpt_path = ckpt_dir / f"reliability_aware_CDM_seed{seed}_{stage}_best.pt"
    encoders = Encoders(vocabs, cfg).to(device)
    model = build_model("reliability_aware", encoders, len(cfg["classes"]["names"]), cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def run_inference(model, loader, device, num_classes):
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            for k in ["clinical_img", "dermoscopy_img", "availability", "label"]:
                batch[k] = batch[k].to(device)
            for k in batch["metadata"]:
                batch["metadata"][k] = batch["metadata"][k].to(device)
            out = model(batch, active_letters=("C", "D", "M"))
            probs = torch.softmax(out["logits"], dim=-1).cpu().numpy()
            all_probs.append(probs)
            all_preds.append(probs.argmax(axis=1))
            all_labels.append(batch["label"].cpu().numpy())
    return (np.concatenate(all_labels), np.concatenate(all_preds), np.concatenate(all_probs))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto",
                     help="'auto' (cuda if available else cpu), or an explicit torch device "
                          "string e.g. 'cuda:0', 'cuda:1', 'cpu'.")
    ap.add_argument("--config", default=CONFIG_PATH)
    args = ap.parse_args()

    config_dir = Path(__file__).parent / Path(args.config).parent
    cfg = load_config(Path(__file__).parent / args.config)
    paths = resolve_data_paths(cfg, config_dir)
    class_names = cfg["classes"]["names"]
    num_classes = len(class_names)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    lesions_all = load_lesion_table(cfg, paths)
    vocabs = build_categorical_vocabs(lesions_all, cfg["metadata_fields"]["categorical"])

    splits_dir = Path(__file__).parent / cfg["split"]["out_dir"]

    report_lines = ["# Outlier Analysis: seed2/A5 and seed3/Full vs. seed0 reference\n"]
    results = {}

    for seed, stage, label in RUNS_TO_ANALYZE:
        print(f"\n=== {label} ===")
        val_csv = splits_dir / f"seed{seed}_val.csv"
        val_ds = MILK10kDataset(val_csv, cfg, "val", vocabs, fixed_modalities={"C", "D", "M"})
        val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False,
                                 num_workers=0)

        model, ckpt = load_model_for_run(seed, stage, cfg, paths, vocabs, device)
        y_true, y_pred, y_probs = run_inference(model, val_loader, device, num_classes)

        mf1 = macro_f1(y_true, y_pred, num_classes)
        per_class_f1 = f1_score(y_true, y_pred, labels=list(range(num_classes)), average=None,
                                 zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        ece = expected_calibration_error(y_true, y_probs)
        mean_entropy = float(predictive_entropy(y_probs).mean())
        mean_conf = float(confidence(y_probs).mean())
        support = np.bincount(y_true, minlength=num_classes)

        results[label] = {
            "macro_f1_recomputed": mf1, "per_class_f1": per_class_f1.tolist(),
            "support": support.tolist(), "ece": ece, "mean_entropy": mean_entropy,
            "mean_confidence": mean_conf, "confusion_matrix": cm.tolist(),
            "checkpoint_epoch": ckpt.get("epoch"), "checkpoint_val_f1": ckpt.get("val_macro_f1"),
        }

        print(f"Recomputed val macro F1: {mf1:.4f} (checkpoint says {ckpt.get('val_macro_f1'):.4f} "
              f"at epoch {ckpt.get('epoch')})")
        print(f"ECE: {ece:.4f}  mean_entropy: {mean_entropy:.4f}  mean_confidence: {mean_conf:.4f}")
        print("Per-class F1 (class: F1, support):")
        for i, cn in enumerate(class_names):
            print(f"  {cn:>10}: F1={per_class_f1[i]:.3f}  support={support[i]}")

        report_lines.append(f"## {label}\n")
        report_lines.append(f"- Recomputed val macro F1: {mf1:.4f} (checkpoint logged "
                             f"{ckpt.get('val_macro_f1'):.4f} at epoch {ckpt.get('epoch')})")
        report_lines.append(f"- ECE: {ece:.4f}, mean entropy: {mean_entropy:.4f}, "
                             f"mean confidence: {mean_conf:.4f}\n")
        report_lines.append("| Class | F1 | Support |")
        report_lines.append("|---|---|---|")
        for i, cn in enumerate(class_names):
            report_lines.append(f"| {cn} | {per_class_f1[i]:.3f} | {support[i]} |")
        report_lines.append("")
        report_lines.append("Confusion matrix (rows=true, cols=pred, order matches classes above):")
        report_lines.append("```")
        report_lines.append("      " + " ".join(f"{cn[:5]:>6}" for cn in class_names))
        for i, cn in enumerate(class_names):
            report_lines.append(f"{cn[:5]:>5} " + " ".join(f"{v:>6}" for v in cm[i]))
        report_lines.append("```\n")

    out_dir = Path(__file__).parent / cfg["output"]["results_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "outlier_analysis.md"
    with open(md_path, "w") as f:
        f.write("\n".join(report_lines))
    json_path = out_dir / "outlier_analysis.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWritten: {md_path}")
    print(f"Written: {json_path}")


if __name__ == "__main__":
    main()
