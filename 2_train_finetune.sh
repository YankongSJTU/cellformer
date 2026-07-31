#!/bin/bash
# 2_train_finetune.sh — Fine-tune CPSformer on user's data with pre-trained weights
#
# Usage:
#   bash scripts/2_train_finetune.sh --pkl_dir ./output --gpu 0
#   bash scripts/2_train_finetune.sh --pkl_dir ./output --gpu 0 --batch_size 64 --epochs 100

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default arguments
PKL_DIR=""
GPU_ID=0
BATCH_SIZE=64
EPOCHS=100
LR=5e-5
ACCUM_STEPS=1
MAX_CELLS=2500
CHECKPOINTS_DIR="${PROJECT_ROOT}/checkpoints_finetuned"
PRETRAINED_MODEL="${PROJECT_ROOT}/checkpoints/best_model.pth"
DISTILLED_CELL="${PROJECT_ROOT}/checkpoints/checkpoints_cellfeature/model.pth"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --pkl_dir)        PKL_DIR="$2"; shift 2 ;;
        --gpu)            GPU_ID="$2"; shift 2 ;;
        --batch_size)     BATCH_SIZE="$2"; shift 2 ;;
        --epochs)         EPOCHS="$2"; shift 2 ;;
        --lr)             LR="$2"; shift 2 ;;
        --accum_steps)    ACCUM_STEPS="$2"; shift 2 ;;
        --max_cells)      MAX_CELLS="$2"; shift 2 ;;
        --output_dir)     CHECKPOINTS_DIR="$2"; shift 2 ;;
        --pretrained)     PRETRAINED_MODEL="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 --pkl_dir DIR [options]"
            echo ""
            echo "Arguments:"
            echo "  --pkl_dir       Directory containing per-cohort PKL files (required)"
            echo "  --gpu           GPU device ID (default: 0)"
            echo "  --batch_size    Batch size (default: 64)"
            echo "  --epochs        Number of epochs (default: 100)"
            echo "  --lr            Learning rate (default: 5e-5)"
            echo "  --accum_steps   Gradient accumulation steps (default: 1)"
            echo "  --max_cells     Max cells per bag (default: 2500)"
            echo "  --output_dir    Output directory for checkpoints"
            echo "  --pretrained    Path to pre-trained model weights"
            exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Validate
if [ -z "$PKL_DIR" ]; then
    echo "Error: --pkl_dir is required."
    echo "Run with --help for usage."
    exit 1
fi

if [ ! -d "$PKL_DIR" ]; then
    echo "Error: PKL directory '$PKL_DIR' does not exist."
    exit 1
fi

mkdir -p "$CHECKPOINTS_DIR"

echo "=== CPSformer Fine-tuning ==="
echo "PKL directory: ${PKL_DIR}"
echo "GPU:           ${GPU_ID}"
echo "Batch size:    ${BATCH_SIZE}"
echo "Accum steps:   ${ACCUM_STEPS}"
echo "Effective batch: $(($BATCH_SIZE * $ACCUM_STEPS))"
echo "Epochs:        ${EPOCHS}"
echo "Learning rate: ${LR}"
echo "Max cells:     ${MAX_CELLS}"
echo "Output:        ${CHECKPOINTS_DIR}"
echo "Pretrained:    ${PRETRAINED_MODEL}"
echo ""

# Check for pre-trained weights
if [ ! -f "$PRETRAINED_MODEL" ]; then
    echo "Warning: Pre-trained model not found at ${PRETRAINED_MODEL}"
    echo "Download from: [PLACEHOLDER_LINK]"
    echo "Training will proceed without loading pre-trained weights."
    PRETRAINED_ARG=""
else
    PRETRAINED_ARG="--pretrained_model_path ${PRETRAINED_MODEL}"
fi

if [ ! -f "$DISTILLED_CELL" ]; then
    echo "Warning: Distilled cell encoder not found at ${DISTILLED_CELL}"
    echo "Download from: [PLACEHOLDER_LINK]"
fi

echo "Starting training..."
echo ""

CUDA_VISIBLE_DEVICES=$GPU_ID python "${PROJECT_ROOT}/train_single_cohort.py" \
    --pkl_dir "$PKL_DIR" \
    --checkpoints_dir "$CHECKPOINTS_DIR" \
    --distilled_cell_path "$DISTILLED_CELL" \
    --epoch_count $EPOCHS \
    --batch_size $BATCH_SIZE \
    --accum_steps $ACCUM_STEPS \
    --max_cells $MAX_CELLS \
    --lr $LR \
    --gradient_checkpointing \
    --encoder_chunk_size 32000 \
    $PRETRAINED_ARG \
    --gpu_id 0

echo ""
echo "=== Training complete! ==="
echo "Best model saved to: ${CHECKPOINTS_DIR}/best_model.pth"
echo "Training log: ${CHECKPOINTS_DIR}/train_log.csv"