"""
Phase 8 prep: dump per-lesion predictions (probs + true label + demographics) for the Full-stage
reliability_aware model, across all 7 modality configs and all 5 seeds, on the TEST split.

This is READ-ONLY inference over existing checkpoints -- same pattern as evaluate.py, just saving
raw per-sample output instead of aggregate metrics. These per-sample predictions are the input to
bootstrap_ci.py (lesion-level bootstrap CI) and subgroup_analysis.py (sex/age/skin-tone breakdown)
-- both need per-lesion rows, which evaluate.py's aggregate-only CSV doesn't have.

Only the Full ablation stage is dumped (the complete proposed model, all 5 components) -- A1..A5
are ablation variants, not separate models we need per-sample predictions for here.

Resume-safe: skips (seed, config) pairs already present in the output CSV.

Usage (from src/):
    python save_predictions.py --device cuda:0 --seeds 0,2,4 --csv_suffix _gpu0
    python save_predictions.py --device cuda:1 --seeds 1,3   --csv_suffix _gpu1

Output: ../outputs/results/predictions_full_stage<suffix>.csv
    one row per (seed, modality_config, lesion_id): true/pred label, per-class probs, demographics.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import load_config, resolve_data_paths, load_lesion_table
from dataset import MILK10kDataset, build_categorical_vocabs
from models import Encoders, build_model

SEEDS_DEFAULT = [0, 1, 2, 3, 4]
STAGE = "Full"


def parse_list_arg(s, cast=str):
    return [cast(x) for x in s.split(",")] if s else None


def load_existing_keys(csv_path):
    if not csv_path.exists():
        return set()
    keys = set()
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            keys.add((int(row["seed"]), row["modality_config"]))
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--config", default="../configs/config.yaml")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--eval_batch_size", type=int, default=32)
    ap.add_argument("--csv_suffix", default="",
                     help="use different suffixes when running on two GPUs at once, same reason "
                          "as evaluate.py -- avoids two processes appending to the same file "
                          "concurrently. Merge afterward (disjoint by seed).")
    args = ap.parse_args()

    seeds = parse_list_arg(args.seeds, int) or SEEDS_DEFAULT

    config_dir = Path(__file__).parent / Path(args.config).parent
    cfg = load_config(Path(__file__).parent / args.config)
    paths = resolve_data_paths(cfg, config_dir)
    num_classes = len(cfg["classes"]["names"])
    modality_configs = [set(mc) for mc in cfg["modality_configs"]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" \
        else torch.device(args.device)
    print(f"Using device: {device}")

    lesions_all = load_lesion_table(cfg, paths)
    vocabs = build_categorical_vocabs(lesions_all, cfg["metadata_fields"]["categorical"])

    ckpt_dir = Path(__file__).parent / cfg["output"]["checkpoints_dir"]
    results_dir = Path(__file__).parent / cfg["output"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"predictions_full_stage{args.csv_suffix}.csv"
    done = load_existing_keys(csv_path)
    print(f"{len(done)} (seed, config) cells already in {csv_path.name}, will be skipped.")

    fieldnames = (["seed", "modality_config", "lesion_id", "label_true", "label_pred",
                    "sex", "age_approx", "skin_tone_class"] +
                  [f"prob_{i}" for i in range(num_classes)])
    is_new = not csv_path.exists()
    csv_file = open(csv_path, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if is_new:
        writer.writeheader()

    splits_dir = Path(__file__).parent / cfg["split"]["out_dir"]
    total_cells = len(seeds) * len(modality_configs)
    done_count = 0

    for seed in seeds:
        split_csv = splits_dir / f"seed{seed}_{args.split}.csv"
        assert split_csv.exists(), f"Missing split file {split_csv}"
        demo_df = __import__("pandas").read_csv(split_csv).set_index("lesion_id")

        ckpt_path = ckpt_dir / f"reliability_aware_CDM_seed{seed}_{STAGE}_best.pt"
        model = None

        loaders = {}
        for mc in modality_configs:
            key = ",".join(sorted(mc))
            ds = MILK10kDataset(split_csv, cfg, args.split, vocabs, fixed_modalities=mc)
            loaders[key] = DataLoader(ds, batch_size=args.eval_batch_size, shuffle=False,
                                       num_workers=0)

        for mc in modality_configs:
            config_key = ",".join(sorted(mc))
            done_count += 1
            if (seed, config_key) in done:
                continue
            if not ckpt_path.exists():
                print(f"SKIP seed{seed}: checkpoint not found at {ckpt_path}")
                break

            if model is None:
                encoders = Encoders(vocabs, cfg).to(device)
                model = build_model("reliability_aware", encoders, num_classes, cfg).to(device)
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model_state"])
                model.eval()

            print(f"[{done_count}/{total_cells}] seed{seed} config={config_key} ...",
                  end=" ", flush=True)

            n_written = 0
            with torch.no_grad():
                for batch in loaders[config_key]:
                    for k in ["clinical_img", "dermoscopy_img", "availability", "label"]:
                        batch[k] = batch[k].to(device)
                    for k in batch["metadata"]:
                        batch["metadata"][k] = batch["metadata"][k].to(device)
                    active = tuple(m for i, m in enumerate(["C", "D", "M"])
                                    if batch["availability"][0, i] > 0)
                    out = model(batch, active_letters=active)
                    probs = torch.softmax(out["logits"], dim=-1).cpu().numpy()
                    preds = probs.argmax(axis=1)
                    labels = batch["label"].cpu().numpy()
                    for i, lid in enumerate(batch["lesion_id"]):
                        demo = demo_df.loc[lid]
                        row = {
                            "seed": seed, "modality_config": config_key, "lesion_id": lid,
                            "label_true": int(labels[i]), "label_pred": int(preds[i]),
                            "sex": demo["sex"], "age_approx": demo["age_approx"],
                            "skin_tone_class": demo["skin_tone_class"],
                        }
                        for k in range(num_classes):
                            row[f"prob_{k}"] = float(probs[i, k])
                        writer.writerow(row)
                        n_written += 1
            csv_file.flush()
            print(f"wrote {n_written} rows")

    csv_file.close()
    print(f"\nDone. Predictions in {csv_path}")


if __name__ == "__main__":
    main()
