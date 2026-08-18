#!/usr/bin/env bash
# ============================================================
#  scripts/download_data.sh
#  UA-LDG: Dataset setup guide
#
#  Primary dataset: MIMIC-CXR-JPG v2.0.0 (you already have this)
#  Optional:        NIH ChestX-ray14, Stanford CheXpert
#
#  Run from the project root:
#    bash scripts/download_data.sh
# ============================================================

set -e

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  UA-LDG: Dataset Setup Guide"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

mkdir -p data/raw/mimic-cxr-jpg
mkdir -p data/raw/chestxray14
mkdir -p data/raw/chexpert
mkdir -p data/processed

# ── 1. MIMIC-CXR-JPG v2.0.0 (PRIMARY DATASET) ───────────────────────────────
echo "── 1. MIMIC-CXR-JPG v2.0.0 (PRIMARY — required) ──────────"
echo ""
echo "  If you already have this dataset, ensure it is placed at:"
echo "    data/raw/mimic-cxr-jpg/"
echo ""
echo "  Required files and structure:"
echo "    data/raw/mimic-cxr-jpg/"
echo "    ├── files/"
echo "    │   ├── p10/"
echo "    │   │   └── p10000032/"
echo "    │   │       └── s50414267/"
echo "    │   │           └── <dicom_id>.jpg"
echo "    │   └── p11/ ... p19/"
echo "    ├── mimic-cxr-2.0.0-split.csv.gz"
echo "    ├── mimic-cxr-2.0.0-chexpert.csv.gz"
echo "    ├── mimic-cxr-2.0.0-metadata.csv.gz"
echo "    └── mimic-cxr-2.0.0-negbio.csv.gz   (not used, but may be present)"
echo ""
echo "  If you need to download it:"
echo "    1. Register at https://physionet.org (requires credentialing)"
echo "    2. Request access to MIMIC-CXR-JPG:"
echo "       https://physionet.org/content/mimic-cxr-jpg/"
echo "    3. On Bridges-2, use wget with your PhysioNet credentials:"
echo "       wget -r -N -c -np --user YOUR_PHYSIONET_USER \\"
echo "            --ask-password \\"
echo "            https://physionet.org/files/mimic-cxr-jpg/2.0.0/ \\"
echo "            -P data/raw/mimic-cxr-jpg/"
echo ""

# ── 2. Verify MIMIC-CXR-JPG placement ────────────────────────────────────────
echo "── 2. Verifying MIMIC-CXR-JPG files ──────────────────────"
echo ""

MIMIC_ROOT="data/raw/mimic-cxr-jpg"
REQUIRED_FILES=(
    "$MIMIC_ROOT/mimic-cxr-2.0.0-split.csv.gz"
    "$MIMIC_ROOT/mimic-cxr-2.0.0-chexpert.csv.gz"
    "$MIMIC_ROOT/mimic-cxr-2.0.0-metadata.csv.gz"
)

ALL_FOUND=true
for f in "${REQUIRED_FILES[@]}"; do
    # Also accept .csv (uncompressed)
    if [ -f "$f" ] || [ -f "${f%.gz}" ]; then
        echo "  ✓ Found: $f"
    else
        echo "  ✗ Missing: $f"
        ALL_FOUND=false
    fi
done

if [ -d "$MIMIC_ROOT/files" ]; then
    N_IMAGES=$(find "$MIMIC_ROOT/files" -name "*.jpg" | wc -l)
    echo "  ✓ Found files/ directory ($N_IMAGES JPG images)"
else
    echo "  ✗ Missing: $MIMIC_ROOT/files/"
    ALL_FOUND=false
fi

if [ "$ALL_FOUND" = true ]; then
    echo ""
    echo "  ✓ MIMIC-CXR-JPG is ready. Proceed to preprocessing."
else
    echo ""
    echo "  Some files are missing. Place MIMIC-CXR-JPG under $MIMIC_ROOT"
    echo "  before running preprocess.py."
fi
echo ""

# ── 3. BioWordVec embeddings ──────────────────────────────────────────────────
echo "── 3. BioWordVec Clinical Word Embeddings ─────────────────"
echo ""

BIOVEC_URL="https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioWordVec_PubMed_MIMICIII_d200.vec.bin"
BIOVEC_PATH="data/processed/biowordvec_200d.bin"

if [ -f "$BIOVEC_PATH" ]; then
    echo "  ✓ BioWordVec already present at $BIOVEC_PATH"
else
    echo "  Required for high-quality label embeddings (~1.7 GB)."
    echo "  URL: $BIOVEC_URL"
    echo ""
    echo "  On Bridges-2 (run from login node):"
    echo "    wget -c -O $BIOVEC_PATH \\"
    echo "         $BIOVEC_URL"
    echo ""
    read -r -p "  Download now? [y/N] " response
    if [[ "$response" =~ ^([yY])$ ]]; then
        wget -c -O "$BIOVEC_PATH" "$BIOVEC_URL"
        echo "  ✓ BioWordVec downloaded."
    else
        echo "  Skipped. preprocess.py will use random embeddings as fallback."
        echo "  Replace before final experiments."
    fi
fi
echo ""

# ── 4. Optional: NIH ChestX-ray14 ────────────────────────────────────────────
echo "── 4. NIH ChestX-ray14 (optional — cross-dataset eval) ────"
echo ""
echo "  Not required for training. Used only if you want cross-dataset"
echo "  evaluation or to improve S_cross edge confidence estimation."
echo ""
echo "  If needed:"
echo "    1. Visit: https://nihcc.app.box.com/v/ChestXray-NIHCC"
echo "    2. Download images_001.tar.gz ... images_012.tar.gz"
echo "    3. Download Data_Entry_2017.csv, train_val_list.txt, test_list.txt"
echo "    4. Extract images to: data/raw/chestxray14/images/"
echo "    5. Place CSVs in:     data/raw/chestxray14/"
echo ""

# ── 5. Next steps ─────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Next steps:"
echo ""
echo "  Local machine:"
echo "    python scripts/preprocess.py --config configs/default.yaml"
echo "    python scripts/train.py --config configs/default.yaml"
echo ""
echo "  Bridges-2 (PSC):"
echo "    sbatch scripts/slurm_preprocess.sh"
echo "    sbatch scripts/slurm_train.sh"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
