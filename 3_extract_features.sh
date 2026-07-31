#!/bin/bash
# 3_extract_features.sh — Extract CPS features from ROI images
# Automatically runs nucleus segmentation if no masks are provided.
#
# Usage:
#   bash scripts/3_extract_features.sh --input_dir ./my_data --gpu 0
#   bash scripts/3_extract_features.sh --input_dir ./my_data --model_path ./checkpoints/best_model.pth --gpu 0

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default arguments
INPUT_DIR=""
OUTPUT_DIR=""
GPU_ID=0
MODEL_PATH="${PROJECT_ROOT}/checkpoints/best_model.pth"
CELL_ENCODER="${PROJECT_ROOT}/checkpoints/checkpoints_cellfeature/model.pth"
MAX_CELLS=2500

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input_dir)    INPUT_DIR="$2"; shift 2 ;;
        --output_dir)   OUTPUT_DIR="$2"; shift 2 ;;
        --gpu)          GPU_ID="$2"; shift 2 ;;
        --model_path)   MODEL_PATH="$2"; shift 2 ;;
        --cell_encoder) CELL_ENCODER="$2"; shift 2 ;;
        --max_cells)    MAX_CELLS="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 --input_dir DIR [--model_path PATH] [--gpu ID] [options]"
            echo ""
            echo "Arguments:"
            echo "  --input_dir    Directory with cohort subdirectories containing ROI images"
            echo "  --output_dir   Directory for extracted CSV features (default: features/)"
            echo "  --gpu           GPU device ID (default: 0)"
            echo "  --model_path    Path to CPSformer model checkpoint"
            echo "  --cell_encoder  Path to distilled cell encoder checkpoint"
            echo "  --max_cells     Max cells per image (default: 2500)"
            echo ""
            echo "Input format:"
            echo "  ${INPUT_DIR}/"
            echo "    ├── COHORT_A/"
            echo "    │   ├── image1.png"
            echo "    │   └── segment/  (optional: pre-computed masks)"
            echo "    └── COHORT_B/"
            echo ""
            echo "Output: {cohort}.cps_feature.csv with 1024-dim features per patch"
            exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Validate
if [ -z "$INPUT_DIR" ]; then
    echo "Error: --input_dir is required."
    echo "Run with --help for usage."
    exit 1
fi

if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' does not exist."
    exit 1
fi

OUTPUT_DIR="${OUTPUT_DIR:-${INPUT_DIR}/features}"
mkdir -p "$OUTPUT_DIR"

echo "=== CPSformer Feature Extraction ==="
echo "Input:      ${INPUT_DIR}"
echo "Output:     ${OUTPUT_DIR}"
echo "GPU:        ${GPU_ID}"
echo "Model:      ${MODEL_PATH}"
echo ""

# Check model
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model checkpoint not found at ${MODEL_PATH}"
    echo "Download pre-trained weights from: [PLACEHOLDER_LINK]"
    exit 1
fi

if [ ! -f "$CELL_ENCODER" ]; then
    echo "Error: Cell encoder not found at ${CELL_ENCODER}"
    echo "Download from: [PLACEHOLDER_LINK]"
    exit 1
fi

# Check segmentation: does each cohort have segment/ subdirs?
echo "Checking for pre-computed segmentation masks..."
HAS_SEG=true
for cohort_dir in "$INPUT_DIR"/*/; do
    cohort_name=$(basename "$cohort_dir")
    if [ ! -d "${cohort_dir}segment" ]; then
        echo "  ${cohort_name}: No segment/ directory — will run auto-segmentation"
        HAS_SEG=false
    fi
done

if [ "$HAS_SEG" = "true" ]; then
    echo "  All cohorts have pre-computed masks. Skipping segmentation."
fi

echo ""
echo "Extracting features..."

# For each cohort directory
for cohort_dir in "$INPUT_DIR"/*/; do
    [ -d "$cohort_dir" ] || continue
    cohort_name=$(basename "$cohort_dir")

    # Skip if segment/ is the only subdirectory
    if [ "$cohort_name" = "segment" ]; then
        continue
    fi

    echo ""
    echo "--- Processing: ${cohort_name} ---"

    # Auto-segmentation if needed
    if [ ! -d "${cohort_dir}segment" ]; then
        echo "  Running automatic nucleus segmentation..."
        mkdir -p "${cohort_dir}segment"
        # Collect image paths and pass as positional arguments
        IMG_ARGS=()
        for img in "$cohort_dir"/*.{png,jpg,jpeg,tif}; do
            [ -f "$img" ] && IMG_ARGS+=("$img")
        done
        if [ ${#IMG_ARGS[@]} -eq 0 ]; then
            echo "  No images found in ${cohort_dir}, skipping."
            continue
        fi
        CUDA_VISIBLE_DEVICES=$GPU_ID python "${PROJECT_ROOT}/nucseg_modules/nucseg_deeplabv3.py" \
            "${IMG_ARGS[@]}" \
            --work_dir "${cohort_dir}.nucseg_tmp" \
            --gpu_id 0 \
            --output_dir "${cohort_dir}segment"
        echo "  Segmentation complete."
    fi

    # Extract features
    CUDA_VISIBLE_DEVICES=$GPU_ID python "${PROJECT_ROOT}/extract_cps_features.py" \
        --root_dir "$INPUT_DIR" \
        --cohort "$cohort_name" \
        --model_path "$MODEL_PATH" \
        --distilled_cell_path "$CELL_ENCODER" \
        --max_cells "$MAX_CELLS" \
        --output_dir "$OUTPUT_DIR" \
        --auto_segment

    echo "  Features saved to: ${OUTPUT_DIR}/${cohort_name}.cps_feature.csv"
done

echo ""
echo "=== Feature extraction complete! ==="
echo "Output directory: ${OUTPUT_DIR}"
echo ""
echo "Next step:"
echo "  bash scripts/4_run_downstream.sh --features_dir ${OUTPUT_DIR} --clinical_dir ./clinical_data"