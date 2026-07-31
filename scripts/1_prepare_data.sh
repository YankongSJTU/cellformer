#!/bin/bash
# 1_prepare_data.sh — End-to-end data preparation: ROI images → PKL
# Supports automatic nucleus segmentation if no masks are provided.
#
# Usage:
#   bash scripts/1_prepare_data.sh --input_dir ./my_data --output_dir ./output --gpu 0
#   bash scripts/1_prepare_data.sh --input_dir ./my_data --output_dir ./output --gpu 0 --skip_seg

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default arguments
INPUT_DIR=""
OUTPUT_DIR=""
GPU_ID=0
SKIP_SEG=false
MAX_CELLS=2500
PATCH_SIZE=1000

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input_dir)  INPUT_DIR="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --gpu)         GPU_ID="$2"; shift 2 ;;
        --skip_seg)    SKIP_SEG=true; shift 1 ;;
        --max_cells)   MAX_CELLS="$2"; shift 2 ;;
        --patch_size)  PATCH_SIZE="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 --input_dir DIR --output_dir DIR [--gpu ID] [--skip_seg] [--max_cells N]"
            echo ""
            echo "Arguments:"
            echo "  --input_dir   Directory containing cohort subdirectories with ROI images (.png/.jpg)"
            echo "  --output_dir  Directory to save generated PKL files"
            echo "  --gpu          GPU device ID (default: 0)"
            echo "  --skip_seg     Skip nucleus segmentation (use existing masks in segment/ subdirs)"
            echo "  --max_cells    Max cells per image (default: 2500)"
            echo "  --patch_size   Patch size for segmentation (default: 1000)"
            exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Validate arguments
if [ -z "$INPUT_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Error: --input_dir and --output_dir are required."
    echo "Run with --help for usage."
    exit 1
fi

if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' does not exist."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "=== CPSformer Data Preparation ==="
echo "Input:  ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "GPU:    ${GPU_ID}"
echo "Skip segmentation: ${SKIP_SEG}"
echo ""

# Check directory structure
echo "Expected input structure:"
echo "  ${INPUT_DIR}/"
echo "    ├── COHORT_A/"
echo "    │   ├── image1.png"
echo "    │   ├── image2.png"
echo "    │   └── segment/        (optional: pre-computed masks)"
echo "    │       ├── image1.png"
echo "    │       └── image2.png"
echo "    └── COHORT_B/"
echo "        └── ..."
echo ""

# Count images
TOTAL_IMAGES=$(find "$INPUT_DIR" -maxdepth 2 \( -name "*.png" -o -name "*.jpg" -o -name "*.tif" \) ! -path "*/segment/*" | wc -l)
echo "Found ${TOTAL_IMAGES} ROI images (excluding segment/ directories)."
echo ""

# Run prepare_data.py
echo "Running data preparation pipeline..."
CUDA_VISIBLE_DEVICES=$GPU_ID python "${PROJECT_ROOT}/prepare_data.py" \
    --input_dir "$INPUT_DIR" \
    --save_path "${OUTPUT_DIR}/train_data.pkl" \
    --gpu_id $GPU_ID \
    --max_cells $MAX_CELLS \
    ${SKIP_SEG:+--skip_segmentation}

echo ""
echo "=== Data preparation complete! ==="
echo "Output PKL: ${OUTPUT_DIR}/train_data.pkl"
echo ""
echo "Next step:"
echo "  bash scripts/2_train_finetune.sh --pkl_dir ${OUTPUT_DIR} --gpu ${GPU_ID}"
