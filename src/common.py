"""Shared utilities: config loading, metadata loading/joining, path resolution."""
import yaml
from pathlib import Path
import pandas as pd


def load_config(config_path="../configs/config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_data_paths(cfg, config_dir: Path):
    """Resolve all data paths relative to the directory containing config.yaml."""
    data_root = (config_dir / cfg["data"]["data_root"]).resolve()
    d = cfg["data"]
    return {
        "data_root": data_root,
        "train_input_dir": data_root / d["train_input_dir"],
        "test_input_dir": data_root / d["test_input_dir"],
        "train_metadata_csv": data_root / d["train_metadata_csv"],
        "train_groundtruth_csv": data_root / d["train_groundtruth_csv"],
        "train_supplement_csv": data_root / d["train_supplement_csv"],
        "test_metadata_csv": data_root / d["test_metadata_csv"],
    }


def load_lesion_table(cfg, paths):
    """
    Build one row per LESION (not per image) with:
      lesion_id, label (int, index into cfg classes.names), label_name,
      clinical_isic_id, clinical_path, dermoscopy_isic_id, dermoscopy_path,
      age_approx, sex, site, skin_tone_class, and MONET_* columns for each of the two images
      (suffixed _clinical / _dermoscopy).
    Lesions missing either image are dropped with a printed warning (should not happen in a
    clean download, but the loader must not silently misalign clinical/dermoscopy pairs).
    """
    meta = pd.read_csv(paths["train_metadata_csv"])
    gt = pd.read_csv(paths["train_groundtruth_csv"])
    class_names = cfg["classes"]["names"]

    # label = argmax over one-hot GT columns
    gt = gt.copy()
    gt["label"] = gt[class_names].values.argmax(axis=1)
    gt["label_name"] = gt["label"].apply(lambda i: class_names[i])
    # sanity: exactly one class should be 1 per row
    row_sums = gt[class_names].sum(axis=1)
    bad = gt[row_sums != 1]
    if len(bad) > 0:
        print(f"WARNING: {len(bad)} lesions in GroundTruth do not have exactly one positive "
              f"class label. Inspect MILK10k_Training_GroundTruth.csv rows: "
              f"{bad['lesion_id'].tolist()[:10]}")

    clinical = meta[meta["image_type"] == "clinical: close-up"].copy()
    dermo = meta[meta["image_type"] == "dermoscopic"].copy()

    monet_cols = cfg["metadata_fields"]["monet_columns"]
    keep_cols = ["lesion_id", "isic_id", "age_approx", "sex", "site", "skin_tone_class"] + monet_cols

    clinical = clinical[keep_cols].rename(
        columns={"isic_id": "clinical_isic_id", **{c: f"{c}_clinical" for c in monet_cols}})
    dermo = dermo[keep_cols].rename(
        columns={"isic_id": "dermoscopy_isic_id", **{c: f"{c}_dermoscopy" for c in monet_cols}})
    # drop duplicated demographic cols from dermo side (identical per lesion; keep clinical's copy)
    dermo = dermo.drop(columns=["age_approx", "sex", "site", "skin_tone_class"])

    lesions = gt[["lesion_id", "label", "label_name"]].merge(
        clinical, on="lesion_id", how="left").merge(
        dermo, on="lesion_id", how="left")

    missing = lesions[lesions["clinical_isic_id"].isna() | lesions["dermoscopy_isic_id"].isna()]
    if len(missing) > 0:
        print(f"WARNING: dropping {len(missing)} lesions missing a clinical or dermoscopy image: "
              f"{missing['lesion_id'].tolist()[:10]}")
        lesions = lesions.dropna(subset=["clinical_isic_id", "dermoscopy_isic_id"])

    lesions["clinical_path"] = lesions.apply(
        lambda r: str(paths["train_input_dir"] / r["lesion_id"] / f"{r['clinical_isic_id']}.jpg"),
        axis=1)
    lesions["dermoscopy_path"] = lesions.apply(
        lambda r: str(paths["train_input_dir"] / r["lesion_id"] / f"{r['dermoscopy_isic_id']}.jpg"),
        axis=1)

    return lesions.reset_index(drop=True)
