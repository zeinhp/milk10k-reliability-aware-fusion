"""
Phase 6: controlled contradictory-evidence stress test (CLAUDE.md "Controlled Contradictory-
Evidence Stress Test", test set only). Swap ONE modality's evidence across lesions while keeping
the other two matched to the original lesion, e.g. (C_i, D_i, M_j) for a metadata contradiction.

Two separate controlled conditions, reported SEPARATELY (never mixed into one number):
  1. same_class swap  -- j has the SAME diagnosis label as i (identity/context inconsistency at
     constant diagnosis: does the model over-trust one modality when another is merely "different
     person, same disease"?)
  2. cross_class swap -- j has a DIFFERENT diagnosis label than i (genuine diagnostic contradiction)

Run for all 3 swap targets (which modality gets replaced: C, D, or M) x both swap types x all 5
seeds x ALL 6 ablation stages (A1..Full, --stages, default all). Running every stage -- not just
Full -- is the point of this version: it lets us isolate WHICH ablation component (if any) makes
the model more/less robust to a contradictory modality, e.g. does A2 (+ reliability weighting)
reduce the damage from a contradictory D image relative to A1 (no reliability weighting at all)?
The first version of this script only tested Full and found Full itself is NOT robust to D
contradiction -- this answers whether that's true of the whole ablation family or specific to it.

This is a SYNTHETIC stress test -- explicitly labeled as such in every output file, per protocol.
Compare against evaluate.py's sweep CSV (per seed x stage, C+D+M config, no-swap) for the
no-contradiction baseline at each stage.

This is READ-ONLY inference over existing checkpoints -- does not train or modify anything.
Resume-safe: skips (seed, stage, swap_modality, swap_type) combos already in the output CSV.

Usage (from src/):
    python stress_test.py --device cuda:0 --seeds 0,2,4 --csv_suffix _gpu0
    python stress_test.py --device cuda:1 --seeds 1,3   --csv_suffix _gpu1
    python stress_test.py --device cuda:0 --seeds 0 --stages A1,Full   # partial, e.g. to test

Output: ../outputs/results/stress_test_contradiction_all_stages<suffix>.csv
"""
import argparse
import csv
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from common import load_config, resolve_data_paths, load_lesion_table
from dataset import MILK10kDataset, build_categorical_vocabs, get_transforms, MISSING_TOKEN
from models import Encoders, build_model
from metrics import macro_f1, balanced_accuracy

SEEDS_DEFAULT = [0, 1, 2, 3, 4]
STAGES_DEFAULT = ["A1", "A2", "A3", "A4", "A5", "Full"]
SWAP_MODALITIES = ["C", "D", "M"]
SWAP_TYPES = ["same_class", "cross_class"]

CSV_FIELDS = ["seed", "ablation_stage", "swap_modality", "swap_type", "n_samples", "macro_f1",
              "balanced_acc"]


def parse_list_arg(s, cast=str):
    return [cast(x) for x in s.split(",")] if s else None


def load_existing_keys(csv_path):
    if not csv_path.exists():
        return set()
    keys = set()
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            keys.add((int(row["seed"]), row["ablation_stage"], row["swap_modality"], row["swap_type"]))
    return keys


def append_row(csv_path, row):
    is_new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if is_new:
            w.writeheader()
        w.writerow(row)


class ContradictionDataset(Dataset):
    """Every item uses all 3 modalities (availability all-ones), but `swap_modality`'s evidence
    for lesion i is taken from a DIFFERENT lesion j, chosen per `swap_type`. The other two
    modalities stay matched to lesion i. Label reported is always lesion i's true label (that's
    what the model SHOULD say if it were robust to the one contradictory modality)."""

    def __init__(self, csv_path, cfg, vocabs, swap_modality, swap_type, pairing_seed=0):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.cfg = cfg
        self.vocabs = vocabs
        self.swap_modality = swap_modality
        self.transform = get_transforms(cfg, "test")

        rng = random.Random(pairing_seed)
        n = len(self.df)
        idx_by_label = {}
        for i, lbl in enumerate(self.df["label"]):
            idx_by_label.setdefault(lbl, []).append(i)

        pair_idx = []
        for i in range(n):
            lbl = self.df.loc[i, "label"]
            if swap_type == "same_class":
                candidates = [j for j in idx_by_label[lbl] if j != i]
            else:
                candidates = [j for j in range(n) if self.df.loc[j, "label"] != lbl]
            if not candidates:
                candidates = [j for j in range(n) if j != i]
            pair_idx.append(rng.choice(candidates))
        self.pair_idx = pair_idx

    def __len__(self):
        return len(self.df)

    def _encode_metadata(self, row):
        age = row["age_approx"]
        age_norm = 0.0 if pd.isna(age) else float(age) / 100.0
        age_missing = 1.0 if pd.isna(age) else 0.0
        cat_ids = {}
        for field in ["sex", "site", "skin_tone_class"]:
            val = row[field]
            vocab = self.vocabs[field]
            cat_ids[field] = vocab[val] if (not pd.isna(val) and val in vocab) else vocab[MISSING_TOKEN]
        return {
            "age_norm": torch.tensor(age_norm, dtype=torch.float32),
            "age_missing": torch.tensor(age_missing, dtype=torch.float32),
            "sex_id": torch.tensor(cat_ids["sex"], dtype=torch.long),
            "site_id": torch.tensor(cat_ids["site"], dtype=torch.long),
            "skin_tone_id": torch.tensor(cat_ids["skin_tone_class"], dtype=torch.long),
        }

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pair_row = self.df.iloc[self.pair_idx[idx]]

        c_src = pair_row if self.swap_modality == "C" else row
        d_src = pair_row if self.swap_modality == "D" else row
        m_src = pair_row if self.swap_modality == "M" else row

        clinical_img = self.transform(Image.open(c_src["clinical_path"]).convert("RGB"))
        dermoscopy_img = self.transform(Image.open(d_src["dermoscopy_path"]).convert("RGB"))
        metadata = self._encode_metadata(m_src)
        availability = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)

        return {
            "clinical_img": clinical_img,
            "dermoscopy_img": dermoscopy_img,
            "metadata": metadata,
            "availability": availability,
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "lesion_id": row["lesion_id"],
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--config", default="../configs/config.yaml")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--stages", default=None, help="comma-separated, default all of A1..Full")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--eval_batch_size", type=int, default=32)
    ap.add_argument("--csv_suffix", default="")
    args = ap.parse_args()

    seeds = parse_list_arg(args.seeds, int) or SEEDS_DEFAULT
    stages = parse_list_arg(args.stages, str) or STAGES_DEFAULT

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
    csv_path = results_dir / f"stress_test_contradiction_all_stages{args.csv_suffix}.csv"
    done = load_existing_keys(csv_path)
    print(f"{len(done)} (seed, stage, swap_modality, swap_type) cells already in {csv_path.name}, "
          f"will be skipped.")

    splits_dir = Path(__file__).parent / cfg["split"]["out_dir"]
    total_cells = len(seeds) * len(stages) * len(SWAP_MODALITIES) * len(SWAP_TYPES)
    done_count = 0

    for seed in seeds:
        split_csv = splits_dir / f"seed{seed}_{args.split}.csv"
        assert split_csv.exists(), f"Missing split file {split_csv}"

        # Pairing (which lesion j gets swapped in for lesion i) only depends on seed + swap_type,
        # not on the ablation stage -- build once per seed and reuse across all 6 stages so every
        # stage is compared under the EXACT same contradiction pairs (a fair ablation comparison).
        pairing_cache = {}
        for swap_modality in SWAP_MODALITIES:
            for swap_type in SWAP_TYPES:
                ds = ContradictionDataset(split_csv, cfg, vocabs, swap_modality, swap_type,
                                           pairing_seed=seed)
                pairing_cache[(swap_modality, swap_type)] = DataLoader(
                    ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=0)

        for stage in stages:
            ckpt_path = ckpt_dir / f"reliability_aware_CDM_seed{seed}_{stage}_best.pt"
            model = None

            for swap_modality in SWAP_MODALITIES:
                for swap_type in SWAP_TYPES:
                    done_count += 1
                    cache_key = (seed, stage, swap_modality, swap_type)
                    if cache_key in done:
                        continue
                    if not ckpt_path.exists():
                        print(f"SKIP seed{seed}/{stage}: checkpoint not found at {ckpt_path}")
                        continue

                    if model is None:
                        encoders = Encoders(vocabs, cfg).to(device)
                        model = build_model("reliability_aware", encoders, num_classes, cfg).to(device)
                        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                        model.load_state_dict(ckpt["model_state"])
                        model.eval()

                    print(f"[{done_count}/{total_cells}] seed{seed} {stage} swap={swap_modality} "
                          f"type={swap_type} ...", end=" ", flush=True)

                    loader = pairing_cache[(swap_modality, swap_type)]

                    all_true, all_pred = [], []
                    with torch.no_grad():
                        for batch in loader:
                            for k in ["clinical_img", "dermoscopy_img", "availability", "label"]:
                                batch[k] = batch[k].to(device)
                            for k in batch["metadata"]:
                                batch["metadata"][k] = batch["metadata"][k].to(device)
                            out = model(batch, active_letters=("C", "D", "M"))
                            probs = torch.softmax(out["logits"], dim=-1).cpu().numpy()
                            all_pred.append(probs.argmax(axis=1))
                            all_true.append(batch["label"].cpu().numpy())
                    y_true = np.concatenate(all_true)
                    y_pred = np.concatenate(all_pred)

                    row = {
                        "seed": seed, "ablation_stage": stage,
                        "swap_modality": swap_modality, "swap_type": swap_type,
                        "n_samples": len(y_true),
                        "macro_f1": macro_f1(y_true, y_pred, num_classes),
                        "balanced_acc": balanced_accuracy(y_true, y_pred),
                    }
                    append_row(csv_path, row)
                    print(f"macro_f1={row['macro_f1']:.4f}")

    print(f"\nDone. Results in {csv_path}")


if __name__ == "__main__":
    main()
