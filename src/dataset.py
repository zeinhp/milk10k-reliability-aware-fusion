"""
PyTorch Dataset for MILK10k multimodal lesion classification.

Handles:
  - loading clinical + dermoscopy images with train/val/test-appropriate transforms
  - metadata encoding inputs (raw values passed through; embedding happens in the model)
  - explicit missing-category handling for categorical metadata
  - stochastic modality dropout for training (arbitrary missing-modality training)
  - fixed modality masking for evaluation (one of the 7 mandatory configs)
"""
import random
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

MODALITY_NAMES = ["C", "D", "M"]

# categorical vocab — built once from the full training metadata by the caller and passed in,
# so val/test never see an unseen category collapse to something undefined.
MISSING_TOKEN = "__MISSING__"


def build_categorical_vocabs(lesions_df, categorical_fields):
    vocabs = {}
    for field in categorical_fields:
        values = sorted(lesions_df[field].dropna().unique().tolist())
        vocabs[field] = {MISSING_TOKEN: 0, **{v: i + 1 for i, v in enumerate(values)}}
    return vocabs


def get_transforms(cfg, split):
    img_cfg = cfg["image"]
    if split == "train":
        return transforms.Compose([
            transforms.Resize((img_cfg["resize"], img_cfg["resize"])),
            transforms.RandomCrop(img_cfg["crop"]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(img_cfg["rotation_deg"]),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=img_cfg["mean"], std=img_cfg["std"]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_cfg["resize"], img_cfg["resize"])),
            transforms.CenterCrop(img_cfg["crop"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=img_cfg["mean"], std=img_cfg["std"]),
        ])


ALL_SUBSETS_NONEMPTY = [
    {"C"}, {"D"}, {"M"}, {"C", "D"}, {"C", "M"}, {"D", "M"}, {"C", "D", "M"},
]


def sample_modality_subset(full_modality_prob: float):
    if random.random() < full_modality_prob:
        return {"C", "D", "M"}
    others = [s for s in ALL_SUBSETS_NONEMPTY if s != {"C", "D", "M"}]
    return random.choice(others)


class MILK10kDataset(Dataset):
    """
    split: "train" | "val" | "test"
    fixed_modalities: if given (e.g. {"C","D"}), every item uses exactly this subset — used for
        evaluation on one of the 7 mandatory configs. If None and split == "train", modality
        dropout is applied stochastically per __getitem__ (arbitrary missing-modality training).
        If None and split != "train", defaults to full {"C","D","M"} (no dropout at eval unless
        fixed_modalities explicitly requests a subset).
    """

    def __init__(self, csv_path, cfg, split, vocabs, fixed_modalities=None):
        self.df = pd.read_csv(csv_path)
        self.cfg = cfg
        self.split = split
        self.vocabs = vocabs
        self.fixed_modalities = fixed_modalities
        self.transform = get_transforms(cfg, split)
        self.full_modality_prob = cfg["training"]["modality_dropout"]["full_modality_prob"]
        self.num_classes = len(cfg["classes"]["names"])

    def __len__(self):
        return len(self.df)

    def _encode_metadata(self, row):
        age = row["age_approx"]
        age_norm = 0.0 if pd.isna(age) else float(age) / 100.0  # simple 0-1ish normalization
        age_missing = 1.0 if pd.isna(age) else 0.0

        cat_ids = {}
        for field in ["sex", "site", "skin_tone_class"]:
            val = row[field]
            vocab = self.vocabs[field]
            if pd.isna(val) or val not in vocab:
                cat_ids[field] = vocab[MISSING_TOKEN]
            else:
                cat_ids[field] = vocab[val]
        return {
            "age_norm": torch.tensor(age_norm, dtype=torch.float32),
            "age_missing": torch.tensor(age_missing, dtype=torch.float32),
            "sex_id": torch.tensor(cat_ids["sex"], dtype=torch.long),
            "site_id": torch.tensor(cat_ids["site"], dtype=torch.long),
            "skin_tone_id": torch.tensor(cat_ids["skin_tone_class"], dtype=torch.long),
        }

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if self.fixed_modalities is not None:
            active = self.fixed_modalities
        elif self.split == "train":
            active = sample_modality_subset(self.full_modality_prob)
        else:
            active = {"C", "D", "M"}

        clinical_img = torch.zeros(3, self.cfg["image"]["crop"], self.cfg["image"]["crop"])
        dermoscopy_img = torch.zeros(3, self.cfg["image"]["crop"], self.cfg["image"]["crop"])

        if "C" in active:
            clinical_img = self.transform(Image.open(row["clinical_path"]).convert("RGB"))
        if "D" in active:
            dermoscopy_img = self.transform(Image.open(row["dermoscopy_path"]).convert("RGB"))

        metadata = self._encode_metadata(row) if "M" in active else self._encode_metadata(
            pd.Series({"age_approx": None, "sex": None, "site": None, "skin_tone_class": None}))

        availability = torch.tensor(
            [1.0 if m in active else 0.0 for m in MODALITY_NAMES], dtype=torch.float32)

        return {
            "clinical_img": clinical_img,
            "dermoscopy_img": dermoscopy_img,
            "metadata": metadata,
            "availability": availability,  # order [C, D, M]
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "lesion_id": row["lesion_id"],
        }


def modality_letters_to_availability(letters):
    """e.g. {'C','D'} -> [1,1,0] following MODALITY_NAMES order."""
    return [1.0 if m in letters else 0.0 for m in MODALITY_NAMES]
