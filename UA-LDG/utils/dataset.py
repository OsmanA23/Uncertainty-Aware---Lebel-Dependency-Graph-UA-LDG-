"""
utils/dataset.py
────────────────
Dataset classes for multi-label CXR classification.

Supports:
  - MIMICCXRDataset     : MIMIC-CXR-JPG v2.0.0 (377K images, 12 native CheXpert labels)
  - ChestXray14Dataset  : NIH ChestX-ray14 (112K images, 14 labels)
  - CheXpertDataset     : Stanford CheXpert (224K images, mapped to 12 MIMIC labels)

All classes return:
  image    : (3, 224, 224)  normalised tensor
  label    : (12,)          binary float tensor
  index    : int            sample index (for uncertainty / vacuity analysis)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MIMIC-CXR-JPG folder structure (from README):
  files/
    p10/               ← first two chars of subject_id folder name
      p10000032/       ← subject_id
        s50414267/     ← study_id
          <dicom_id>.jpg
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MIMIC-CXR-JPG label values (from README):
   1.0  → positive (pathology present)
   0.0  → negative (pathology absent)
  -1.0  → uncertain (ambiguous or hedged language in report)
   NaN  → not mentioned in the report

Uncertainty policy options for -1.0 / NaN:
  "negative"  → treat as 0  (conservative; most common in literature)
  "positive"  → treat as 1  (liberal upper bound)
  "ignore"    → exclude these samples from loss (requires custom loss masking)
"""

import os
from typing import Optional, Tuple, Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


# ── ImageNet normalisation ────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── MIMIC-CXR-JPG native 12-pathology ordering ───────────────────────────────
# Uses MIMIC-CXR's own CheXpert label names directly.
# Excludes meta-labels: "No Finding" (no pathology) and "Support Devices".
PATHOLOGIES = [
    "Atelectasis",                 # 0
    "Cardiomegaly",                # 1
    "Consolidation",               # 2
    "Edema",                       # 3
    "Enlarged Cardiomediastinum",  # 4
    "Fracture",                    # 5
    "Lung Lesion",                 # 6
    "Lung Opacity",                # 7
    "Pleural Effusion",            # 8
    "Pneumonia",                   # 9
    "Pneumothorax",                # 10
    "Pleural Other",               # 11
]

# ── MIMIC-CXR-JPG CheXpert label names → PATHOLOGIES mapping ─────────────────
# Direct identity mapping — no renaming to NIH equivalents.
# Meta-labels (No Finding, Support Devices) are dropped.
MIMIC_CHEXPERT_TO_STD = {
    "Atelectasis":                "Atelectasis",
    "Cardiomegaly":               "Cardiomegaly",
    "Consolidation":              "Consolidation",
    "Edema":                      "Edema",
    "Enlarged Cardiomediastinum": "Enlarged Cardiomediastinum",
    "Fracture":                   "Fracture",
    "Lung Lesion":                "Lung Lesion",
    "Lung Opacity":               "Lung Opacity",
    "Pleural Effusion":           "Pleural Effusion",
    "Pneumonia":                  "Pneumonia",
    "Pneumothorax":               "Pneumothorax",
    "Pleural Other":              "Pleural Other",
    "Support Devices":            None,   # not a pathology
    "No Finding":                 None,   # meta-label
}

# ── CheXpert label names → PATHOLOGIES mapping (Stanford CheXpert dataset) ────
# Uses MIMIC-CXR native names directly — no NIH aliasing.
CHEXPERT_TO_STD = {
    "Atelectasis":      "Atelectasis",
    "Cardiomegaly":     "Cardiomegaly",
    "Pleural Effusion": "Pleural Effusion",
    "Lung Opacity":     "Lung Opacity",
    "Lung Lesion":      "Lung Lesion",
    "Edema":            "Edema",
    "Consolidation":    "Consolidation",
    "Pneumonia":        "Pneumonia",
    "Pneumothorax":     "Pneumothorax",
    "Pleural Other":    "Pleural Other",
    "Fracture":         "Fracture",
    "No Finding":       None,
    "Support Devices":  None,
}

# ── Co-occurrence threshold ───────────────────────────────────────────────────
# Minimum number of co-occurring patient cases for an edge to exist in the
# disease graph. Edges below this count are excluded entirely.
COOCCURRENCE_THRESHOLD = 100

# ── Frontal view positions to keep (from MIMIC metadata) ─────────────────────
FRONTAL_VIEWS = {"PA", "AP"}


# ── Transforms ───────────────────────────────────────────────────────────────

def get_transforms(split: str, image_size: int = 224) -> transforms.Compose:
    """
    Returns torchvision transforms for train / val / test splits.

    Args:
        split      : "train" | "val" | "test"
        image_size : target square size in pixels (default 224)
    """
    if split == "train":
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


# ── Label uncertainty helpers ─────────────────────────────────────────────────

def _apply_uncertainty_policy(
    val: float,
    policy: str,
) -> float:
    """
    Convert a raw MIMIC/CheXpert label value to a binary float.

    MIMIC-CXR label values:
       1.0  → positive
       0.0  → negative
      -1.0  → uncertain
       NaN  → not mentioned

    Args:
        val    : raw label value
        policy : one of "negative" | "positive" | "ignore"
                 "ignore" returns -1.0 as a sentinel so the loss
                 function can mask these samples out.

    Returns:
        0.0, 1.0, or -1.0 (sentinel for "ignore")
    """
    if pd.isna(val):
        # Not mentioned → treat as negative (standard practice)
        return 0.0
    val = float(val)
    if val == 1.0:
        return 1.0
    if val == 0.0:
        return 0.0
    if val == -1.0:           # uncertain label
        if policy == "negative":
            return 0.0
        elif policy == "positive":
            return 1.0
        elif policy == "ignore":
            return -1.0       # sentinel: exclude from loss
    return 0.0                # fallback


# ══════════════════════════════════════════════════════════════════════════════
# MIMIC-CXR-JPG Dataset
# ══════════════════════════════════════════════════════════════════════════════

class MIMICCXRDataset(Dataset):
    """
    MIMIC-CXR-JPG v2.0.0 dataset loader.

    Handles the nested folder structure, frontal-view filtering,
    official train/validate/test splits, and CheXpert label uncertainty.

    Expected files (all .gz files are auto-decompressed by pandas):
        <mimic_root>/
            files/
                p10/p10000032/s50414267/<dicom_id>.jpg
                ...
            mimic-cxr-2.0.0-split.csv.gz      (or .csv)
            mimic-cxr-2.0.0-chexpert.csv.gz   (or .csv)
            mimic-cxr-2.0.0-metadata.csv.gz   (or .csv)

    Key design decisions (from README):
      1. Labels are per-study; images are per-dicom.
         We join on study_id to assign each image its study's labels.
      2. We filter to frontal views only (PA and AP) using the metadata file.
         Lateral views don't show the same pathological features.
      3. The official split file uses 'validate' (not 'val') — we normalise to 'val'.
      4. Uncertain labels (-1.0) are handled by `uncertain_policy`.
      5. Missing labels (NaN) are always treated as negative (absent).

    Args:
        mimic_root       : root directory of MIMIC-CXR-JPG
                           (contains files/ and the .csv.gz files)
        split            : "train" | "val" | "test"
        split_csv        : path to mimic-cxr-2.0.0-split.csv[.gz]
        label_csv        : path to mimic-cxr-2.0.0-chexpert.csv[.gz]
        metadata_csv     : path to mimic-cxr-2.0.0-metadata.csv[.gz]
        uncertain_policy : how to handle -1.0 labels:
                           "negative" (default) | "positive" | "ignore"
        frontal_only     : if True (default), keep only PA and AP views
        transform        : optional custom transform
        image_size       : resize target (default 224)

    Returns per sample:
        image    : (3, H, W) normalised tensor
        label    : (12,) binary float32 tensor (MIMIC-CXR native PATHOLOGIES ordering)
        index    : int sample index
    """

    def __init__(
        self,
        mimic_root: str,
        split: str = "train",
        split_csv: str = None,
        label_csv: str = None,
        metadata_csv: str = None,
        uncertain_policy: str = "negative",
        frontal_only: bool = True,
        transform=None,
        image_size: int = 224,
        validate_images: bool = False,
    ) -> None:
        assert split in ("train", "val", "test"), \
            f"split must be 'train', 'val', or 'test', got '{split}'"
        assert uncertain_policy in ("negative", "positive", "ignore"), \
            f"uncertain_policy must be 'negative'/'positive'/'ignore'"

        self.mimic_root       = mimic_root
        self.split            = split
        self.uncertain_policy = uncertain_policy
        self.transform        = transform or get_transforms(split, image_size)

        # ── Resolve default CSV paths ──────────────────────────────────────
        def _find_csv(explicit, *candidates):
            """Return explicit path if given, else find first existing candidate."""
            if explicit and os.path.exists(explicit):
                return explicit
            for c in candidates:
                full = os.path.join(mimic_root, c)
                if os.path.exists(full):
                    return full
            raise FileNotFoundError(
                f"Could not find CSV. Tried: {candidates}. "
                f"Set the explicit path argument or check your mimic_root."
            )

        split_csv = _find_csv(
            split_csv,
            "mimic-cxr-2.0.0-split.csv.gz",
            "mimic-cxr-2.0.0-split.csv",
        )
        label_csv = _find_csv(
            label_csv,
            "mimic-cxr-2.0.0-chexpert.csv.gz",
            "mimic-cxr-2.0.0-chexpert.csv",
        )
        metadata_csv = _find_csv(
            metadata_csv,
            "mimic-cxr-2.0.0-metadata.csv.gz",
            "mimic-cxr-2.0.0-metadata.csv",
        )

        # ── Load CSVs ─────────────────────────────────────────────────────
        print(f"[MIMICCXRDataset] Loading split CSV    : {split_csv}")
        df_split = pd.read_csv(split_csv)
        # Official file uses 'validate'; normalise to 'val'
        df_split["split"] = df_split["split"].replace("validate", "val")

        print(f"[MIMICCXRDataset] Loading label CSV    : {label_csv}")
        df_labels = pd.read_csv(label_csv)

        print(f"[MIMICCXRDataset] Loading metadata CSV : {metadata_csv}")
        df_meta = pd.read_csv(
            metadata_csv,
            usecols=["dicom_id", "ViewPosition"],  # only need view info
        )

        # ── Filter to requested split ──────────────────────────────────────
        df_split = df_split[df_split["split"] == split].copy()
        print(f"[MIMICCXRDataset] Split '{split}': {len(df_split):,} dicoms")

        # ── Filter to frontal views ────────────────────────────────────────
        if frontal_only:
            df_meta_frontal = df_meta[
                df_meta["ViewPosition"].isin(FRONTAL_VIEWS)
            ]
            df_split = df_split[
                df_split["dicom_id"].isin(df_meta_frontal["dicom_id"])
            ].copy()
            print(f"[MIMICCXRDataset] After frontal filter: {len(df_split):,} dicoms")

        # ── Join labels onto split (labels are per study_id) ──────────────
        # df_split has: dicom_id, study_id, subject_id, split
        # df_labels has: subject_id, study_id, Atelectasis, Cardiomegaly, ...
        df = df_split.merge(
            df_labels,
            on=["subject_id", "study_id"],
            how="left",
        )
        print(f"[MIMICCXRDataset] After label join: {len(df):,} samples")

        # ── Build standardised (12,) label matrix ─────────────────────────
        label_matrix = np.zeros((len(df), len(PATHOLOGIES)), dtype=np.float32)

        for mimic_col, std_col in MIMIC_CHEXPERT_TO_STD.items():
            if std_col is None:
                continue                          # dropped label
            if mimic_col not in df.columns:
                continue                          # not in this version
            std_idx = PATHOLOGIES.index(std_col)
            label_matrix[:, std_idx] = df[mimic_col].apply(
                lambda v: _apply_uncertainty_policy(v, uncertain_policy)
            ).values

        # ── Build image paths from subject_id / study_id / dicom_id ───────
        # Structure: files/p{prefix}/p{subject_id}/s{study_id}/{dicom_id}.jpg
        # The prefix folder is always the first 3 chars of "p{subject_id}"
        def _build_path(row) -> str:
            subj  = f"p{row['subject_id']}"
            study = f"s{row['study_id']}"
            dcm   = f"{row['dicom_id']}.jpg"
            prefix = subj[:3]                     # e.g. "p10" from "p10000032"
            return os.path.join(
                mimic_root, "files", prefix, subj, study, dcm
            )

        print("[MIMICCXRDataset] Building image paths ...")
        self.img_paths   = df.apply(_build_path, axis=1).tolist()
        self.labels      = label_matrix
        self.df          = df.reset_index(drop=True)

        if validate_images:
            self._filter_missing_and_corrupt_images()

        # ── Uncertainty mask (for "ignore" policy) ─────────────────────────
        # mask[i, j] = 0 means sample i, label j should be excluded from loss
        if uncertain_policy == "ignore":
            self.uncertainty_mask = (self.labels >= 0).astype(np.float32)
        else:
            self.uncertainty_mask = np.ones_like(self.labels)

        print(
            f"[MIMICCXRDataset] Ready: {len(self):,} samples | "
            f"split={split} | policy={uncertain_policy} | "
            f"frontal_only={frontal_only}"
        )

    def _filter_missing_and_corrupt_images(self) -> None:
        """Remove entries with missing or unreadable image files."""
        valid_mask = []
        missing = 0
        corrupt = 0

        for img_path in self.img_paths:
            if not os.path.exists(img_path):
                missing += 1
                valid_mask.append(False)
                continue
            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    img.load()
                valid_mask.append(True)
            except Exception:
                corrupt += 1
                valid_mask.append(False)

        if missing or corrupt:
            print(
                f"[MIMICCXRDataset] Removed {missing:,} missing "
                f"and {corrupt:,} corrupt images from split."
            )

        valid_mask = np.array(valid_mask, dtype=bool)
        if not valid_mask.any():
            raise RuntimeError(
                "No valid MIMIC-CXR images found after validation. "
                "Check your mimic_root and downloaded image files."
            )

        self.img_paths = [p for p, ok in zip(self.img_paths, valid_mask) if ok]
        self.labels    = self.labels[valid_mask]
        self.df        = self.df[valid_mask].reset_index(drop=True)

    # ── Statistics helper ─────────────────────────────────────────────────────
    def label_stats(self) -> pd.DataFrame:
        """
        Return a DataFrame summarising positive rate per pathology.
        Useful for verifying the dataset loaded correctly.
        """
        pos_labels = (self.labels == 1.0)
        rows = []
        for i, p in enumerate(PATHOLOGIES):
            n_pos = pos_labels[:, i].sum()
            rows.append({
                "pathology":    p,
                "n_positive":   int(n_pos),
                "n_total":      len(self),
                "positive_rate": round(n_pos / len(self), 4),
            })
        return pd.DataFrame(rows)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        img_path = self.img_paths[idx]
        try:
            with Image.open(img_path) as img:
                image = img.convert("RGB")
                image.load()
        except (FileNotFoundError, OSError, SyntaxError) as e:
            raise FileNotFoundError(
                f"Could not open image at index {idx}: {img_path}\n"
                f"Check that mimic_root is correct and the files/ directory exists.\n"
                f"Original error: {e}"
            )
        image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label, idx


# ══════════════════════════════════════════════════════════════════════════════
# NIH ChestX-ray14 Dataset
# ══════════════════════════════════════════════════════════════════════════════

class ChestXray14Dataset(Dataset):
    """
    NIH ChestX-ray14 dataset loader.

    Expected directory structure:
        <image_dir>/
            00000001_000.png
            ...
        Data_Entry_2017.csv
        train_val_list.txt
        test_list.txt

    Args:
        image_dir    : path to the flat images/ folder
        label_file   : path to Data_Entry_2017.csv
        split_file   : path to train_val_list.txt or test_list.txt
        split        : "train" | "val" | "test"
        transform    : optional custom transform
        image_size   : resize target
        val_fraction : fraction of train_val set used for validation
        seed         : random seed for reproducible train/val split
    """

    def __init__(
        self,
        image_dir: str,
        label_file: str,
        split_file: str,
        split: str = "train",
        transform=None,
        image_size: int = 224,
        val_fraction: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.image_dir = image_dir
        self.split     = split
        self.transform = transform or get_transforms(split, image_size)

        df = pd.read_csv(label_file)
        df = df[["Image Index", "Finding Labels"]].copy()

        # Parse pipe-separated finding labels into binary columns
        for path in PATHOLOGIES:
            df[path] = df["Finding Labels"].str.contains(path).astype(float)

        # Filter to official split file
        with open(split_file) as f:
            split_files = set(line.strip() for line in f)
        df = df[df["Image Index"].isin(split_files)].reset_index(drop=True)

        # Split train_val → train / val
        if split in ("train", "val") and "test" not in split_file:
            rng   = np.random.default_rng(seed)
            idx   = np.arange(len(df))
            rng.shuffle(idx)
            n_val = int(len(df) * val_fraction)
            if split == "val":
                df = df.iloc[idx[:n_val]].reset_index(drop=True)
            else:
                df = df.iloc[idx[n_val:]].reset_index(drop=True)

        self.df          = df
        self.image_names = df["Image Index"].tolist()
        self.labels      = df[PATHOLOGIES].values.astype(np.float32)
        # NIH labels are clean binary — no uncertainty mask needed
        self.uncertainty_mask = np.ones_like(self.labels)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        img_path = os.path.join(self.image_dir, self.image_names[idx])
        image    = Image.open(img_path).convert("RGB")
        image    = self.transform(image)
        label    = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label, idx


# ══════════════════════════════════════════════════════════════════════════════
# Stanford CheXpert Dataset
# ══════════════════════════════════════════════════════════════════════════════

class CheXpertDataset(Dataset):
    """
    Stanford CheXpert dataset loader.

    Expected directory structure:
        <data_root>/
            train.csv
            valid.csv
            train/  (images)
            valid/  (images)

    Args:
        csv_file         : path to train.csv or valid.csv
        data_root        : root directory (parent of train/ and valid/)
        split            : "train" | "val"
        uncertain_policy : "negative" | "positive" | "ignore"
        transform        : optional custom transform
        image_size       : resize target
    """

    def __init__(
        self,
        csv_file: str,
        data_root: str,
        split: str = "train",
        uncertain_policy: str = "negative",
        transform=None,
        image_size: int = 224,
    ) -> None:
        self.data_root        = data_root
        self.split            = split
        self.uncertain_policy = uncertain_policy
        self.transform        = transform or get_transforms(split, image_size)

        df = pd.read_csv(csv_file)

        label_matrix = np.zeros((len(df), len(PATHOLOGIES)), dtype=np.float32)
        for chex_col, std_col in CHEXPERT_TO_STD.items():
            if std_col is None or chex_col not in df.columns:
                continue
            std_idx = PATHOLOGIES.index(std_col)
            label_matrix[:, std_idx] = df[chex_col].apply(
                lambda v: _apply_uncertainty_policy(v, uncertain_policy)
            ).values

        self.df         = df
        self.img_paths  = df["Path"].tolist()
        self.labels     = label_matrix

        if uncertain_policy == "ignore":
            self.uncertainty_mask = (label_matrix >= 0).astype(np.float32)
        else:
            self.uncertainty_mask = np.ones_like(label_matrix)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        img_path = os.path.join(self.data_root, self.img_paths[idx])
        image    = Image.open(img_path).convert("RGB")
        image    = self.transform(image)
        label    = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label, idx


# ── DataLoader factory ────────────────────────────────────────────────────────

def build_dataloader(
    dataset: Dataset,
    split: str,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    """Convenience wrapper around torch DataLoader."""
    from torch.utils.data import DataLoader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(split == "train"),
        persistent_workers=(num_workers > 0),
    )


# ── Quick dataset inspection ──────────────────────────────────────────────────

def inspect_dataset(dataset, name: str = "Dataset") -> None:
    """
    Print a summary of a loaded dataset — useful for verifying
    that MIMIC-CXR-JPG loaded correctly before submitting a training job.
    """
    print(f"\n{'─'*55}")
    print(f"  {name}")
    print(f"{'─'*55}")
    print(f"  Total samples : {len(dataset):,}")

    if hasattr(dataset, "label_stats"):
        stats = dataset.label_stats()
        print(f"\n  {'Pathology':<25} {'Positive':>10} {'Rate':>8}")
        print(f"  {'─'*43}")
        for _, row in stats.iterrows():
            bar = "█" * int(row["positive_rate"] * 20)
            print(
                f"  {row['pathology']:<25} "
                f"{row['n_positive']:>10,} "
                f"{row['positive_rate']:>7.1%}  {bar}"
            )
    else:
        labels = dataset.labels
        print(f"\n  {'Pathology':<25} {'Positive':>10} {'Rate':>8}")
        print(f"  {'─'*43}")
        for i, p in enumerate(PATHOLOGIES):
            n_pos = (labels[:, i] == 1).sum()
            rate  = n_pos / len(dataset)
            bar   = "█" * int(rate * 20)
            print(f"  {p:<25} {n_pos:>10,} {rate:>7.1%}  {bar}")
    print()
