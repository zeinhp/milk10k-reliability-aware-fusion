"""
Unified training script. Trains ONE (model_family, modality_config, seed) combination per run,
so the full experimental matrix (4 baseline families x 7 configs x 5 seeds, + proposed model
variants) is driven by looping this script — see run_all.py for the driver.

Usage (from src/):
    python train.py --family unimodal --modality C --seed 0
    python train.py --family gated_fusion --modality C,D,M --seed 0
    python train.py --family reliability_aware --modality C,D,M --seed 0 --ablation Full

Notes:
  - `--modality` fixes evaluation-time (and, for non-"reliability_aware" families, training-time)
    modality availability to that exact subset, per one of the 7 mandatory configs.
  - Only `reliability_aware` uses stochastic modality dropout during training by default
    (arbitrary missing-modality training, CLAUDE.md section "Arbitrary Missing-Modality
    Training"); other families are trained and evaluated on their fixed --modality config, which
    is how the "train 7 separate models" comparison point is produced for the missing-modality
    robustness table (Section 19 of CLAUDE.md).
  - `--ablation` (only meaningful for reliability_aware) selects one of A1..Full from
    configs/config.yaml's ablation.stages block, toggling reliability_weighting / modality_dropout
    / disagreement / calibration / abstention. Calibration and abstention are applied post-hoc in
    calibrate.py / evaluate.py, not inside this training loop.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import load_config, resolve_data_paths, load_lesion_table
from dataset import MILK10kDataset, build_categorical_vocabs, modality_letters_to_availability
from models import Encoders, build_model
from losses import classification_loss, reliability_aware_total_loss
from metrics import macro_f1, balanced_accuracy


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_modality_arg(s):
    letters = set(s.split(","))
    assert letters.issubset({"C", "D", "M"}), f"invalid modality string: {s}"
    return letters


def run_epoch(model, family, loader, device, optimizer, cfg, lambda_r, train=True):
    model.train(train)
    total_loss, n_batches = 0.0, 0
    all_labels, all_preds, all_probs = [], [], []

    for batch in loader:
        for k in ["clinical_img", "dermoscopy_img", "availability", "label"]:
            batch[k] = batch[k].to(device)
        for k in batch["metadata"]:
            batch["metadata"][k] = batch["metadata"][k].to(device)

        active = {"C", "D", "M"} if family in ("naive_concat", "late_fusion", "gated_fusion",
                                                "reliability_aware") else None

        with torch.set_grad_enabled(train):
            if family == "unimodal":
                out = model(batch)
                loss = classification_loss(out["logits"], batch["label"])
            elif family == "reliability_aware":
                out = model(batch, active_letters=("C", "D", "M"))
                loss, _ = reliability_aware_total_loss(
                    out, batch["label"], batch["availability"], lambda_r=lambda_r)
            else:
                out = model(batch)
                loss = classification_loss(out["logits"], batch["label"])

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        probs = torch.softmax(out["logits"], dim=-1).detach().cpu().numpy()
        all_probs.append(probs)
        all_preds.append(probs.argmax(axis=1))
        all_labels.append(batch["label"].cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    num_classes = len(cfg["classes"]["names"])
    return {
        "loss": total_loss / max(n_batches, 1),
        "macro_f1": macro_f1(all_labels, all_preds, num_classes),
        "balanced_acc": balanced_accuracy(all_labels, all_preds),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True,
                     choices=["unimodal", "naive_concat", "late_fusion", "gated_fusion",
                              "reliability_aware"])
    ap.add_argument("--modality", required=True, help="e.g. C or C,D or C,D,M")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--ablation", default="Full", help="only used by reliability_aware")
    ap.add_argument("--config", default="../configs/config.yaml")
    ap.add_argument("--epochs", type=int, default=None, help="override config epochs (for quick tests)")
    ap.add_argument("--device", default="auto",
                     help="'auto' (cuda if available else cpu), or an explicit torch device "
                          "string e.g. 'cuda:0', 'cuda:1', 'cpu'. Use this to pin a run to one "
                          "GPU when splitting the experiment matrix across multiple machines/GPUs.")
    ap.add_argument("--no_resume", action="store_true",
                     help="ignore any existing *_last.pt checkpoint and start this run from "
                          "scratch instead of resuming (default: resume automatically if found).")
    args = ap.parse_args()

    config_dir = Path(__file__).parent / Path(args.config).parent
    cfg = load_config(Path(__file__).parent / args.config)
    paths = resolve_data_paths(cfg, config_dir)

    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    modality_set = parse_modality_arg(args.modality)
    if args.family == "unimodal":
        assert len(modality_set) == 1, "unimodal family requires exactly one modality"

    splits_dir = Path(__file__).parent / cfg["split"]["out_dir"]
    train_csv = splits_dir / f"seed{args.seed}_train.csv"
    val_csv = splits_dir / f"seed{args.seed}_val.csv"
    assert train_csv.exists(), f"Missing split file {train_csv} — run make_splits.py first."

    # Build categorical vocabs from the FULL lesion table (not just train) so val/test categories
    # are always known, but values are still only ever learned from train during fitting.
    lesions_all = load_lesion_table(cfg, paths)
    vocabs = build_categorical_vocabs(lesions_all, cfg["metadata_fields"]["categorical"])

    fixed_modalities = None if args.family == "reliability_aware" else modality_set
    train_ds = MILK10kDataset(train_csv, cfg, "train", vocabs, fixed_modalities=fixed_modalities)
    val_ds = MILK10kDataset(val_csv, cfg, "val", vocabs, fixed_modalities=modality_set)

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True,
                               num_workers=cfg["training"]["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False,
                             num_workers=cfg["training"]["num_workers"])

    encoders = Encoders(vocabs, cfg).to(device)
    modality_for_build = list(modality_set)[0] if args.family == "unimodal" else None
    model = build_model(args.family, encoders, len(cfg["classes"]["names"]), cfg,
                         modality=modality_for_build).to(device)

    ablation_flags = cfg["ablation"]["stages"].get(args.ablation, cfg["ablation"]["stages"]["Full"])
    lambda_r = cfg["training"]["lambda_reliability"] if ablation_flags.get(
        "reliability_weighting", True) else 0.0

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"],
                                   weight_decay=cfg["training"]["weight_decay"])

    epochs = args.epochs or cfg["training"]["epochs"]
    patience = cfg["training"]["early_stopping_patience"]
    best_val_f1, best_epoch, epochs_no_improve = -1.0, -1, 0

    ckpt_dir = Path(__file__).parent / cfg["output"]["checkpoints_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{args.family}_{args.modality.replace(',', '')}_seed{args.seed}"
    if args.family == "reliability_aware":
        run_name += f"_{args.ablation}"
    best_ckpt_path = ckpt_dir / f"{run_name}_best.pt"
    last_ckpt_path = ckpt_dir / f"{run_name}_last.pt"

    log_dir = Path(__file__).parent / cfg["output"]["logs_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_name}.json"
    log = []
    start_epoch = 0

    # Resume support: browser-based remote sessions (e.g. JupyterHub tabs) can disconnect
    # mid-run. If a "_last" checkpoint exists for this exact run_name, pick up where it left off
    # instead of silently retraining from epoch 0 (unless --no_resume is passed).
    if last_ckpt_path.exists() and not args.no_resume:
        ckpt = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_f1 = ckpt["best_val_f1"]
        best_epoch = ckpt["best_epoch"]
        epochs_no_improve = ckpt["epochs_no_improve"]
        if log_path.exists():
            with open(log_path) as f:
                log = json.load(f).get("history", [])
        print(f"[{run_name}] Resuming from epoch {start_epoch} "
              f"(best so far: val macro F1 {best_val_f1:.4f} @ epoch {best_epoch})")

    for epoch in range(start_epoch, epochs):
        train_metrics = run_epoch(model, args.family, train_loader, device, optimizer, cfg,
                                   lambda_r, train=True)
        val_metrics = run_epoch(model, args.family, val_loader, device, optimizer, cfg,
                                 lambda_r, train=False)
        log.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(f"[{run_name}] epoch {epoch}: train_loss={train_metrics['loss']:.4f} "
              f"train_macroF1={train_metrics['macro_f1']:.4f} "
              f"val_loss={val_metrics['loss']:.4f} val_macroF1={val_metrics['macro_f1']:.4f}")

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "val_macro_f1": best_val_f1, "args": vars(args)}, best_ckpt_path)
        else:
            epochs_no_improve += 1

        # Always save a "_last" checkpoint (every epoch) purely for resume purposes — separate
        # from "_best", which is what evaluate.py will load later.
        torch.save({
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "epoch": epoch, "best_val_f1": best_val_f1, "best_epoch": best_epoch,
            "epochs_no_improve": epochs_no_improve, "args": vars(args),
        }, last_ckpt_path)
        with open(log_path, "w") as f:
            json.dump({"run_name": run_name, "best_val_macro_f1": best_val_f1,
                       "best_epoch": best_epoch, "history": log}, f, indent=2)

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (best epoch {best_epoch}, "
                  f"best val macro F1 {best_val_f1:.4f})")
            break

    print(f"Done. Best val macro F1 = {best_val_f1:.4f} at epoch {best_epoch}. "
          f"Checkpoint: {best_ckpt_path}")


if __name__ == "__main__":
    main()
