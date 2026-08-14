"""
tests/test_mimic_dataset.py
───────────────────────────
Unit tests for MIMICCXRDataset.

Uses synthetic in-memory CSV data to test all logic without needing
the real MIMIC-CXR-JPG files on disk. Verifies:
  - Path construction from subject_id / study_id / dicom_id
  - Frontal view filtering (PA/AP vs LATERAL)
  - Uncertainty policy (negative / positive / ignore)
  - Label mapping from CheXpert to 12-pathology MIMIC-CXR native ordering
  - Split filtering (train / val / test)
  - Statistics helper

Run:  pytest tests/test_mimic_dataset.py -v
"""

import sys
import os
import tempfile
import gzip

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset import (
    MIMICCXRDataset, PATHOLOGIES, FRONTAL_VIEWS,
    MIMIC_CHEXPERT_TO_STD, _apply_uncertainty_policy,
)


# ── Synthetic data fixture ─────────────────────────────────────────────────────

def make_synthetic_mimic(tmpdir: str):
    """
    Create minimal synthetic MIMIC-CXR-JPG CSV files in tmpdir.
    Returns the tmpdir path (acts as mimic_root).
    """
    # 6 dicoms: 3 frontal (PA/AP), 1 lateral, for 3 studies, 2 patients
    split_data = pd.DataFrame([
        {"dicom_id": "dcm001", "study_id": 101, "subject_id": 10001, "split": "train"},
        {"dicom_id": "dcm002", "study_id": 101, "subject_id": 10001, "split": "train"},
        {"dicom_id": "dcm003", "study_id": 102, "subject_id": 10001, "split": "validate"},
        {"dicom_id": "dcm004", "study_id": 201, "subject_id": 20002, "split": "train"},
        {"dicom_id": "dcm005", "study_id": 201, "subject_id": 20002, "split": "train"},
        {"dicom_id": "dcm006", "study_id": 202, "subject_id": 20002, "split": "test"},
    ])

    metadata_data = pd.DataFrame([
        {"dicom_id": "dcm001", "ViewPosition": "PA"},
        {"dicom_id": "dcm002", "ViewPosition": "LATERAL"},   # should be filtered
        {"dicom_id": "dcm003", "ViewPosition": "AP"},
        {"dicom_id": "dcm004", "ViewPosition": "AP"},
        {"dicom_id": "dcm005", "ViewPosition": "PA"},
        {"dicom_id": "dcm006", "ViewPosition": "PA"},
    ])

    # Labels per study (CheXpert format)
    label_data = pd.DataFrame([
        {"subject_id": 10001, "study_id": 101,
         "Atelectasis": 1.0, "Cardiomegaly": 0.0, "Pleural Effusion": -1.0,
         "Consolidation": float("nan"), "Edema": 0.0, "Pneumonia": 1.0,
         "Pneumothorax": 0.0, "Lung Opacity": 0.0, "Lung Lesion": 0.0,
         "Enlarged Cardiomediastinum": 0.0, "Fracture": 0.0,
         "Pleural Other": 0.0, "Support Devices": 0.0, "No Finding": 0.0},
        {"subject_id": 10001, "study_id": 102,
         "Atelectasis": 0.0, "Cardiomegaly": 1.0, "Pleural Effusion": 0.0,
         "Consolidation": 1.0, "Edema": 1.0, "Pneumonia": 0.0,
         "Pneumothorax": 0.0, "Lung Opacity": 0.0, "Lung Lesion": 0.0,
         "Enlarged Cardiomediastinum": 0.0, "Fracture": 0.0,
         "Pleural Other": 0.0, "Support Devices": 0.0, "No Finding": 0.0},
        {"subject_id": 20002, "study_id": 201,
         "Atelectasis": 0.0, "Cardiomegaly": 0.0, "Pleural Effusion": 1.0,
         "Consolidation": 0.0, "Edema": -1.0, "Pneumonia": 0.0,
         "Pneumothorax": 1.0, "Lung Opacity": 1.0, "Lung Lesion": 0.0,
         "Enlarged Cardiomediastinum": 0.0, "Fracture": 0.0,
         "Pleural Other": 0.0, "Support Devices": 0.0, "No Finding": 0.0},
        {"subject_id": 20002, "study_id": 202,
         "Atelectasis": 0.0, "Cardiomegaly": 0.0, "Pleural Effusion": 0.0,
         "Consolidation": 0.0, "Edema": 0.0, "Pneumonia": 0.0,
         "Pneumothorax": 0.0, "Lung Opacity": 0.0, "Lung Lesion": 0.0,
         "Enlarged Cardiomediastinum": 0.0, "Fracture": 0.0,
         "Pleural Other": 0.0, "Support Devices": 0.0, "No Finding": 1.0},
    ])

    # Save as plain CSV (MIMICCXRDataset also accepts non-gzipped)
    split_data.to_csv(os.path.join(tmpdir, "mimic-cxr-2.0.0-split.csv"),
                      index=False)
    metadata_data.to_csv(os.path.join(tmpdir, "mimic-cxr-2.0.0-metadata.csv"),
                         index=False)
    label_data.to_csv(os.path.join(tmpdir, "mimic-cxr-2.0.0-chexpert.csv"),
                      index=False)
    return tmpdir


@pytest.fixture(scope="module")
def mimic_root(tmp_path_factory):
    tmpdir = str(tmp_path_factory.mktemp("mimic"))
    return make_synthetic_mimic(tmpdir)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestMIMICCXRDatasetSplits:

    def test_train_split_count_frontal_only(self, mimic_root):
        """Train split has dcm001 (PA), dcm004 (AP), dcm005 (PA) = 3 frontal."""
        ds = MIMICCXRDataset(mimic_root, split="train", frontal_only=True)
        assert len(ds) == 3, f"Expected 3 frontal train samples, got {len(ds)}"

    def test_train_split_count_all_views(self, mimic_root):
        """Train split without filtering: dcm001, dcm002, dcm004, dcm005 = 4."""
        ds = MIMICCXRDataset(mimic_root, split="train", frontal_only=False)
        assert len(ds) == 4, f"Expected 4 train samples (all views), got {len(ds)}"

    def test_val_split_normalised(self, mimic_root):
        """Validate → val normalisation: dcm003 (AP) in val split."""
        ds = MIMICCXRDataset(mimic_root, split="val", frontal_only=True)
        assert len(ds) == 1, f"Expected 1 val sample, got {len(ds)}"

    def test_test_split(self, mimic_root):
        """Test split: dcm006 (PA) = 1 sample."""
        ds = MIMICCXRDataset(mimic_root, split="test", frontal_only=True)
        assert len(ds) == 1, f"Expected 1 test sample, got {len(ds)}"

    def test_invalid_split_raises(self, mimic_root):
        with pytest.raises(AssertionError):
            MIMICCXRDataset(mimic_root, split="invalid")


class TestMIMICCXRFrontalFilter:

    def test_lateral_excluded(self, mimic_root):
        """dcm002 has ViewPosition=LATERAL and must not appear in frontal dataset."""
        ds = MIMICCXRDataset(mimic_root, split="train", frontal_only=True)
        dicom_ids = ds.df["dicom_id"].tolist()
        assert "dcm002" not in dicom_ids, "LATERAL view dcm002 should be excluded"

    def test_pa_ap_included(self, mimic_root):
        """PA and AP views must be included."""
        ds = MIMICCXRDataset(mimic_root, split="train", frontal_only=True)
        dicom_ids = set(ds.df["dicom_id"].tolist())
        assert "dcm001" in dicom_ids, "PA view dcm001 should be included"
        assert "dcm004" in dicom_ids, "AP view dcm004 should be included"


class TestMIMICCXRLabelMapping:

    def test_label_shape(self, mimic_root):
        ds = MIMICCXRDataset(mimic_root, split="train", frontal_only=True)
        assert ds.labels.shape[1] == 12, \
            f"Expected 12 label columns, got {ds.labels.shape[1]}"

    def test_atelectasis_positive(self, mimic_root):
        """study 101 has Atelectasis=1.0; dcm001 (PA) is in train."""
        ds = MIMICCXRDataset(mimic_root, split="train",
                             frontal_only=True, uncertain_policy="negative")
        # Find the row for dcm001
        idx = ds.df[ds.df["dicom_id"] == "dcm001"].index[0]
        atel_idx = PATHOLOGIES.index("Atelectasis")
        assert ds.labels[idx, atel_idx] == 1.0, \
            "Atelectasis should be 1.0 for dcm001"

    def test_pleural_effusion_mapped(self, mimic_root):
        """'Pleural Effusion' in MIMIC maps directly to 'Pleural Effusion'."""
        ds = MIMICCXRDataset(mimic_root, split="train",
                             frontal_only=True, uncertain_policy="positive")
        eff_idx = PATHOLOGIES.index("Pleural Effusion")
        # dcm004/dcm005 belong to study 201 which has Pleural Effusion=1.0
        idx = ds.df[ds.df["dicom_id"] == "dcm004"].index[0]
        assert ds.labels[idx, eff_idx] == 1.0, \
            "Pleural Effusion=1.0 should map to Pleural Effusion=1.0"

    def test_pneumothorax_positive(self, mimic_root):
        """study 201 has Pneumothorax=1.0."""
        ds = MIMICCXRDataset(mimic_root, split="train",
                             frontal_only=True, uncertain_policy="negative")
        ptx_idx = PATHOLOGIES.index("Pneumothorax")
        idx = ds.df[ds.df["dicom_id"] == "dcm004"].index[0]
        assert ds.labels[idx, ptx_idx] == 1.0

    def test_no_finding_not_in_pathologies(self, mimic_root):
        """'No Finding' should be dropped — it's not a pathology."""
        assert "No Finding" not in PATHOLOGIES

    def test_meta_labels_not_in_pathologies(self, mimic_root):
        """Support Devices and No Finding are meta-labels and must not be in PATHOLOGIES."""
        assert "No Finding" not in PATHOLOGIES
        assert "Support Devices" not in PATHOLOGIES

    def test_old_nih_classes_not_in_pathologies(self, mimic_root):
        """Old NIH-only classes must not appear in the 12-class MIMIC list."""
        for label in ["Emphysema", "Fibrosis", "Hernia", "Nodule",
                      "Effusion", "Infiltration", "Mass", "Pleural_Thickening"]:
            assert label not in PATHOLOGIES, \
                f"'{label}' is an old NIH alias and should not be in PATHOLOGIES"


class TestUncertaintyPolicy:

    def test_negative_policy(self, mimic_root):
        """With policy='negative', uncertain labels (-1.0) become 0.0."""
        ds = MIMICCXRDataset(mimic_root, split="train",
                             frontal_only=True, uncertain_policy="negative")
        # study 101 has Pleural Effusion=-1.0; dcm001 is in train
        eff_idx = PATHOLOGIES.index("Pleural Effusion")
        idx = ds.df[ds.df["dicom_id"] == "dcm001"].index[0]
        assert ds.labels[idx, eff_idx] == 0.0, \
            "uncertain -1.0 should become 0.0 with 'negative' policy"

    def test_positive_policy(self, mimic_root):
        """With policy='positive', uncertain labels (-1.0) become 1.0."""
        ds = MIMICCXRDataset(mimic_root, split="train",
                             frontal_only=True, uncertain_policy="positive")
        eff_idx = PATHOLOGIES.index("Pleural Effusion")
        idx = ds.df[ds.df["dicom_id"] == "dcm001"].index[0]
        assert ds.labels[idx, eff_idx] == 1.0, \
            "uncertain -1.0 should become 1.0 with 'positive' policy"

    def test_ignore_policy_sentinel(self, mimic_root):
        """With policy='ignore', uncertain labels become -1.0 sentinel."""
        ds = MIMICCXRDataset(mimic_root, split="train",
                             frontal_only=True, uncertain_policy="ignore")
        eff_idx = PATHOLOGIES.index("Pleural Effusion")
        idx = ds.df[ds.df["dicom_id"] == "dcm001"].index[0]
        assert ds.labels[idx, eff_idx] == -1.0, \
            "uncertain -1.0 should stay -1.0 as sentinel with 'ignore' policy"

    def test_ignore_policy_uncertainty_mask(self, mimic_root):
        """With policy='ignore', uncertainty_mask should be 0 for sentinel positions."""
        ds = MIMICCXRDataset(mimic_root, split="train",
                             frontal_only=True, uncertain_policy="ignore")
        eff_idx = PATHOLOGIES.index("Pleural Effusion")
        idx = ds.df[ds.df["dicom_id"] == "dcm001"].index[0]
        assert ds.uncertainty_mask[idx, eff_idx] == 0.0, \
            "uncertainty_mask should be 0 for uncertain labels under 'ignore' policy"

    def test_nan_always_negative(self, mimic_root):
        """NaN (not mentioned) is always treated as 0 regardless of policy."""
        for policy in ["negative", "positive", "ignore"]:
            assert _apply_uncertainty_policy(float("nan"), policy) == 0.0, \
                f"NaN should always map to 0.0, failed for policy='{policy}'"

    def test_invalid_policy_raises(self, mimic_root):
        with pytest.raises(AssertionError):
            MIMICCXRDataset(mimic_root, split="train", uncertain_policy="wrong")


class TestMIMICCXRPathConstruction:

    def test_path_structure(self, mimic_root):
        """
        Verify image paths follow:
            files/{p_prefix}/{p_subject}/{s_study}/{dicom_id}.jpg
        """
        ds = MIMICCXRDataset(mimic_root, split="train", frontal_only=True)
        for path in ds.img_paths:
            rel   = os.path.relpath(path, mimic_root)
            parts = rel.split(os.sep)
            assert parts[0] == "files", f"Expected 'files/' prefix, got {parts[0]}"
            assert parts[1].startswith("p"), f"Prefix folder should start with 'p': {parts[1]}"
            assert parts[2].startswith("p"), f"Subject folder should start with 'p': {parts[2]}"
            assert parts[3].startswith("s"), f"Study folder should start with 's': {parts[3]}"
            assert parts[4].endswith(".jpg"), f"Filename should end with .jpg: {parts[4]}"

    def test_prefix_folder_correct(self, mimic_root):
        """Prefix folder = first 3 chars of 'p{subject_id}'."""
        ds = MIMICCXRDataset(mimic_root, split="train", frontal_only=True)
        for i, row in ds.df.iterrows():
            path = ds.img_paths[i]
            expected_prefix = f"p{str(row['subject_id'])[:2]}"
            assert os.sep + expected_prefix + os.sep in path, \
                f"Path {path} missing expected prefix {expected_prefix}"


class TestMIMICCXRStats:

    def test_label_stats_returns_dataframe(self, mimic_root):
        ds = MIMICCXRDataset(mimic_root, split="train", frontal_only=True)
        stats = ds.label_stats()
        assert isinstance(stats, pd.DataFrame)
        assert "pathology" in stats.columns
        assert "positive_rate" in stats.columns
        assert len(stats) == 12

    def test_label_stats_rates_in_range(self, mimic_root):
        ds = MIMICCXRDataset(mimic_root, split="train", frontal_only=True)
        stats = ds.label_stats()
        assert (stats["positive_rate"] >= 0).all()
        assert (stats["positive_rate"] <= 1).all()
