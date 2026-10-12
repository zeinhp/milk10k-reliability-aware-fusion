"""
Phase 6-7 significance-test prep: dump per-lesion entropy, disagreement, and correctness for the
Full-stage reliability_aware model, across all 7 modality configs and 5 seeds, on the TEST split.

evaluate.py already computes aggregate AURC / error-detection AUROC for "entropy alone" vs
"entropy+disagreement combined" risk scores per (seed, stage, modality_config) cell -- but only
the aggregate numbers, not per-lesion risk scores/correctness. This script saves those per-lesion
values so a paired bootstrap can test whether the combined risk score's AURC/error-AUROC advantage
over entropy alone is actually statistically significant, or within noise -- same rigor level
already applied to the baseline-vs-reliability F1/ECE comparison.

Resume-safe: skips (seed, config) pairs already present in the output CSV.

Usage (from src/):
    python save_risk_scores.py --device cuda:0

Output: ../outputs/results/risk_scores_full_stage.csv
    one row per (seed, modality_config, lesion_id): entropy, disagreement (nan if <2 modalities
    active), correct (0/1).
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
from metrics import predictive_entropy, pairwise_js_divergence

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
    ap.add_argument("--csv_suffix", default="")
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
    csv_path = results_dir / f"risk_scores_full_stage{args.csv_suffix}.csv"
    done = load_existing_keys(csv_path)
    print(f"{len(done)} (seed, config) cells already in {csv_path.name}, will be skipped.")

    fieldnames = ["seed", "modality_config", "lesion_id", "entropy", "disagreement", "correct"]
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

        ckpt_path = ckpt_dir / f"reliability_aware_CDM_seed{seed}_{STAGE}_best.pt"
        model = None

        loaders = {}
        for mc in modality_configs:
            key = ",".join(sorted(mc))
            ds = MILK10kDataset(split_csv, cfg, args.split, vocabs, fixed_modalities=mc)
            loaders[key] = DataLoader(ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=0)

        for mc in modality_configs:
            config_key = ",".join(sorted(mc))
            active_mods = sorted(mc)
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

            print(f"[{done_count}/{total_cells}] seed{seed} config={config_key} ...", end=" ", flush=True)

            lesion_ids, all_probs, all_labels = [], [], []
            per_modality_all = {m: [] for m in active_mods}
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
                    all_probs.append(probs)
                    all_labels.append(batch["label"].cpu().numpy())
                    lesion_ids.extend(batch["lesion_id"])
                    for m in active_mods:
                        per_modality_all[m].append(out["per_modality_probs"][m].detach().cpu().numpy())

            y_probs = np.concatenate(all_probs)
            y_true = np.concatenate(all_labels)
            preds = y_probs.argmax(axis=1)
            correct = (preds == y_true).astype(int)
            entropy = predictive_entropy(y_probs)

            if len(active_mods) >= 2:
                pm = {m: np.concatenate(v) for m, v in per_modality_all.items()}
                pair_js = []
                for m1, m2 in itertools.combinations(active_mods, 2):
                    pair_js.append(pairwise_js_divergence(pm[m1], pm[m2]))
                disagreement = np.mean(np.stack(pair_js, axis=0), axis=0)
            else:
                disagreement = np.full(len(y_true), np.nan)

            n_written = 0
            for i, lid in enumerate(lesion_ids):
                writer.writerow({
                    "seed": seed, "modality_config": config_key, "lesion_id": lid,
                    "entropy": float(entropy[i]),
                    "disagreement": float(disagreement[i]) if not np.isnan(disagreement[i]) else "",
                    "correct": int(correct[i]),
                })
                n_written += 1
            csv_file.flush()
            print(f"wrote {n_written} rows")

    csv_file.close()
    print(f"\nDone. Risk scores in {csv_path}")


if __name__ == "__main__":
    main()
