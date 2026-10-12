"""
Phase 4 significance-test prep: dump per-lesion predictions (probs + true label + demographics)
for all 75 baseline checkpoints (B1 unimodal x3 configs, B2/B3/B4 x4 multi-modality configs each),
on the TEST split, at each checkpoint's own trained modality_config. Mirrors save_predictions.py's
pattern for the reliability_aware model, just looped over (family, modality_config) instead of a
single family's 7-config sweep -- see evaluate_baselines.py for why each baseline checkpoint is
only evaluated at its own trained config.

These per-lesion predictions are the input to significance_test.py, which needs row-aligned
per-lesion predictions (not just evaluate_baselines.py's aggregate metrics) to run a paired
bootstrap comparison against the reliability_aware model's predictions
(predictions_full_stage*.csv) on the exact same lesions.

Resume-safe: skips (family, modality_config, seed) triples already present in the output CSV.

Usage (from src/):
    python save_predictions_baselines.py --device cuda:0

Output: ../outputs/results/predictions_baselines<suffix>.csv
    one row per (family, modality_config, seed, lesion_id): true/pred label, per-class probs, demographics.
"""
import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from common import load_config, resolve_data_paths, load_lesion_table
from dataset import MILK10kDataset, build_categorical_vocabs
from models import Encoders, build_model

SEEDS_DEFAULT = [0, 1, 2, 3, 4]
UNIMODAL_CONFIGS = ["C", "D", "M"]
MULTI_FAMILIES = ["naive_concat", "late_fusion", "gated_fusion"]
MULTI_CONFIGS = ["C,D", "C,M", "D,M", "C,D,M"]


def parse_list_arg(s, cast=str):
    return [cast(x) for x in s.split(",")] if s else None


def load_existing_keys(csv_path):
    if not csv_path.exists():
        return set()
    keys = set()
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            keys.add((row["family"], row["modality_config"], int(row["seed"])))
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" \
        else torch.device(args.device)
    print(f"Using device: {device}")

    lesions_all = load_lesion_table(cfg, paths)
    vocabs = build_categorical_vocabs(lesions_all, cfg["metadata_fields"]["categorical"])

    ckpt_dir = Path(__file__).parent / cfg["output"]["checkpoints_dir"]
    results_dir = Path(__file__).parent / cfg["output"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"predictions_baselines{args.csv_suffix}.csv"
    done = load_existing_keys(csv_path)
    print(f"{len(done)} (family, modality_config, seed) cells already in {csv_path.name}, will be skipped.")

    fieldnames = (["family", "modality_config", "seed", "lesion_id", "label_true", "label_pred",
                    "sex", "age_approx", "skin_tone_class"] +
                  [f"prob_{i}" for i in range(num_classes)])
    is_new = not csv_path.exists()
    csv_file = open(csv_path, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if is_new:
        writer.writeheader()

    splits_dir = Path(__file__).parent / cfg["split"]["out_dir"]

    cells = [("unimodal", mc) for mc in UNIMODAL_CONFIGS]
    for family in MULTI_FAMILIES:
        for mc in MULTI_CONFIGS:
            cells.append((family, mc))
    total = len(cells) * len(seeds)
    done_count = 0

    import pandas as pd

    for seed in seeds:
        split_csv = splits_dir / f"seed{seed}_{args.split}.csv"
        assert split_csv.exists(), f"Missing split file {split_csv}"
        demo_df = pd.read_csv(split_csv).set_index("lesion_id")

        loaders = {}
        for mc in UNIMODAL_CONFIGS + MULTI_CONFIGS:
            mc_set = set(mc.split(","))
            ds = MILK10kDataset(split_csv, cfg, args.split, vocabs, fixed_modalities=mc_set)
            loaders[mc] = DataLoader(ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=0)

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

            n_written = 0
            with torch.no_grad():
                for batch in loaders[mc]:
                    for k in ["clinical_img", "dermoscopy_img", "availability", "label"]:
                        batch[k] = batch[k].to(device)
                    for k in batch["metadata"]:
                        batch["metadata"][k] = batch["metadata"][k].to(device)
                    if family == "unimodal":
                        out = model(batch)
                    else:
                        active = tuple(m for i, m in enumerate(["C", "D", "M"])
                                        if batch["availability"][0, i] > 0)
                        out = model(batch, active_letters=active)
                    probs = torch.softmax(out["logits"], dim=-1).cpu().numpy()
                    preds = probs.argmax(axis=1)
                    labels = batch["label"].cpu().numpy()
                    for i, lid in enumerate(batch["lesion_id"]):
                        demo = demo_df.loc[lid]
                        row = {
                            "family": family, "modality_config": mc, "seed": seed, "lesion_id": lid,
                            "label_true": int(labels[i]), "label_pred": int(preds[i]),
                            "sex": demo["sex"], "age_approx": demo["age_approx"],
                            "skin_tone_class": demo["skin_tone_class"],
                        }
                        for k in range(num_classes):
                            row[f"prob_{k}"] = float(probs[i, k])
                        writer.writerow(row)
                        n_written += 1
            csv_file.flush()
            print(f"[{done_count}/{total}] {family}/{mc}/seed{seed}: wrote {n_written} rows", flush=True)

            del model, encoders
            if device.type == "cuda":
                torch.cuda.empty_cache()

    csv_file.close()
    print(f"\nDone. Predictions in {csv_path}")


if __name__ == "__main__":
    main()
