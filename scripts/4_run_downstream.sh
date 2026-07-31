#!/bin/bash
# 4_run_downstream.sh — Run downstream evaluation tasks on extracted CPS features
#
# Tasks:
#   - survival_os: Overall survival (Cox model)
#   - survival_dss: Disease-specific survival (Cox model)
#   - mutation: Gene mutation status prediction
#   - drug: Drug sensitivity prediction (IC50)
#   - msi: MSI status prediction (CRC only: COAD + READ)
#   - tnm: TNM staging prediction
#
# Usage:
#   bash scripts/4_run_downstream.sh --features_dir ./features --clinical_dir ./clinical
#   bash scripts/4_run_downstream.sh --features_dir ./features --task survival_os --gpu 0

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default arguments
FEATURES_DIR=""
CLINICAL_DIR="${PROJECT_ROOT}/clinical"
OUTPUT_DIR=""
GPU_ID=0
TASK="all"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --features_dir)  FEATURES_DIR="$2"; shift 2 ;;
        --clinical_dir)  CLINICAL_DIR="$2"; shift 2 ;;
        --output_dir)    OUTPUT_DIR="$2"; shift 2 ;;
        --gpu)           GPU_ID="$2"; shift 2 ;;
        --task)          TASK="$2"; shift 2 ;;
        -h|--help)
            cat << 'EOF'
Usage: bash scripts/4_run_downstream.sh --features_dir DIR [options]

Required:
  --features_dir   Directory with {cohort}.cps_feature.csv files

Options:
  --clinical_dir   Directory with clinical data (default: ./clinical)
  --output_dir     Output directory for results (default: ./results)
  --gpu            GPU device ID (default: 0)
  --task           Task to run (default: all)
                   Options: all, survival_os, survival_dss, mutation, drug, msi, tnm

Clinical data structure:
  clinical/
  ├── survival/            # OS survival data
  │   └── {cohort}.survival.csv (samplename, time, status)
  ├── survival_dss/        # DSS survival data
  │   └── {cohort}.dss.survival.csv
  ├── mutation/
  │   └── mutationData/{cohort}.all (gene x patient matrix)
  ├── drug/
  │   └── drug.csv (patient, drug_IC50)
  ├── msi/
  │   └── COADREAD.info (MSI labels for CRC)
  └── tnm/
      └── {cohort}_tnm.csv or {cohort}_gdc_clinical.csv

Examples:
  bash scripts/4_run_downstream.sh --features_dir ./features --gpu 0
  bash scripts/4_run_downstream.sh --features_dir ./features --task msi
EOF
            exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Validate
if [ -z "$FEATURES_DIR" ]; then
    echo "Error: --features_dir is required."
    echo "Run with --help for usage."
    exit 1
fi

if [ ! -d "$FEATURES_DIR" ]; then
    echo "Error: Features directory '$FEATURES_DIR' does not exist."
    exit 1
fi

OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/results}"
mkdir -p "$OUTPUT_DIR"

echo "=== CPSformer Downstream Evaluation ==="
echo "Features:  ${FEATURES_DIR}"
echo "Clinical:  ${CLINICAL_DIR}"
echo "Output:    ${OUTPUT_DIR}"
echo "Task:       ${TASK}"
echo "GPU:        ${GPU_ID}"
echo ""

# List available cohort features
echo "Available cohort features:"
ls "$FEATURES_DIR"/*.csv 2>/dev/null | xargs -n1 basename 2>/dev/null | \
    sed 's/\.cps_feature\.csv//' | head -20 || echo "  (none found)"
echo ""

run_task() {
    local task_name="$1"
    local script="$2"
    shift 2
    local extra_args="$@"

    local out_subdir="${OUTPUT_DIR}/results_${task_name}"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Running: ${task_name}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    mkdir -p "$out_subdir"

    if CUDA_VISIBLE_DEVICES=$GPU_ID python "${PROJECT_ROOT}/${script}" \
        --features_dir "${FEATURES_DIR}" \
        --output_dir "$out_subdir" \
        $extra_args; then
        echo "✓ ${task_name} complete → ${out_subdir}"
    else
        echo "✗ ${task_name} failed (check clinical data availability)"
    fi
    echo ""
}

# Run tasks
case "$TASK" in
    all)
        run_task "survival_os" "downstream_survival.py" \
            "--survival_dir ${CLINICAL_DIR}/survival"

        run_task "survival_dss" "downstream_survival.py" \
            "--survival_dir ${CLINICAL_DIR}/survival_dss"

        run_task "mutation" "downstream_mutation_improved.py" \
            "--mutation_dir ${CLINICAL_DIR}/mutation/mutationData"

        run_task "drug" "downstream_drug_improved.py" \
            "--drug_csv ${CLINICAL_DIR}/drug/drug.csv"

        run_task "msi" "downstream_msi.py" \
            "--msi_file ${CLINICAL_DIR}/msi/COADREAD.info"

        run_task "tnm" "downstream_tnm.py" \
            "--clinical_dir ${CLINICAL_DIR}/tnm"
        ;;

    survival_os)
        run_task "survival_os" "downstream_survival.py" \
            "--survival_dir ${CLINICAL_DIR}/survival"
        ;;

    survival_dss)
        run_task "survival_dss" "downstream_survival.py" \
            "--survival_dir ${CLINICAL_DIR}/survival_dss"
        ;;

    mutation)
        run_task "mutation" "downstream_mutation_improved.py" \
            "--mutation_dir ${CLINICAL_DIR}/mutation/mutationData"
        ;;

    drug)
        run_task "drug" "downstream_drug_improved.py" \
            "--drug_csv ${CLINICAL_DIR}/drug/drug.csv"
        ;;

    msi)
        run_task "msi" "downstream_msi.py" \
            "--msi_file ${CLINICAL_DIR}/msi/COADREAD.info"
        ;;

    tnm)
        run_task "tnm" "downstream_tnm.py" \
            "--clinical_dir ${CLINICAL_DIR}/tnm"
        ;;

    *)
        echo "Error: Unknown task '$TASK'"
        echo "Choose from: all, survival_os, survival_dss, mutation, drug, msi, tnm"
        exit 1
        ;;
esac

echo "=== All downstream tasks complete! ==="
echo "Results saved to: ${OUTPUT_DIR}/"