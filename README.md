# CPSformer: Cell Patch Set Transformer for Pathology Image Analysis

[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c37.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-BSD%203--orange.svg)](LICENSE)

**CPSformer** (Cell Patch Set Transformer) is a graph-based framework for whole-slide image (WSI) analysis in computational pathology. It models pathology images as **cell-level graphs** — where each cell is a node with visual and spatial features, and edges capture cell-cell spatial relationships — enabling interpretable, structure-aware representation learning for cancer diagnosis, prognosis prediction, and treatment response.

## Overview

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│  ROI Image   │───▶│   Nucleus    │───▶│   Cell-level│───▶│  CPSformer   │
│  (H&E patch) │    │ Segmentation │    │   Graph +    │    │  Feature     │
│              │    │  (Auto/Custom)│    │   Transformer│    │  (1024-dim)  │
└─────────────┘    └──────────────┘    └─────────────┘    └──────┬───────┘
                                                           │
              ┌────────────────┬────────────────┬─────────┴──────────┐
              ▼                ▼                ▼                    ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐
        │ Survival │  │ Mutation│  │   Drug   │  │ TNM Staging │
        │ Prognosis│  │  Status  │  │Response  │  │ Gleason Grading│
        └──────────┘  └──────────┘  └──────────┘  └──────────────┘
```

### Key Features

- **Cell-level representation**: Each cell is explicitly modeled as a graph node with visual features (from a distilled foundation model) and spatial coordinates
- **Two-scale spatial graph**: Combines local KNN neighbors (fine-grained) with global connections (context-aware) via attention-weighted GAT
- **Dual pooling**: Mean pooling + sparse-attention query pooling for complementary bag-level representations
- **Cross-scale contrastive pre-training**: Full-patch view vs. random subgraph crop augmentation for scale-invariant representations
- **Automatic nucleus segmentation**: Users only need to provide ROI images — nucleus segmentation runs automatically with built-in DeepLabV3 + UNet ensemble
- **Customizable segmentation**: Users can provide their own segmentation masks or replace the segmentation model entirely

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Demo Data](#demo-data)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Inference / Feature Extraction](#inference--feature-extraction)
- [Downstream Tasks](#downstream-tasks)
- [Advanced: WSI-Level Analysis](#advanced-wsi-level-analysis)
- [Pre-trained Models](#pre-trained-models)
- [Project Structure](#project-structure)
- [Performance Benchmark](#performance-benchmark)
- [Citation](#citation)
- [License](#license)

## Installation

### Prerequisites
- Python >= 3.10
- PyTorch >= 1.12 with CUDA support
- NVIDIA GPU (recommended; CPU mode possible but slow)

### Setup

```bash
git clone https://github.com/YankongSJTU/CPSformer.git
cd CPSformer

# Option 1: One-click environment setup
bash scripts/setup_env.sh

# Option 2: Manual setup
conda create -n cpsformer python=3.10 -y
conda activate cpsformer
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
pip install -r requirements.txt
```

## Quick Start

```bash
# 1. Download pre-trained models and demo data
#    See [Demo Data](#demo-data) and [Pre-trained Models](#pre-trained-models) sections

# 2. Prepare data (auto segmentation included)
bash scripts/1_prepare_data.sh \
    --input_dir ./demo/TCGA \
    --output_dir ./demo/prepared \
    --gpu 0

# 3. Fine-tune on your data
bash scripts/2_train_finetune.sh \
    --pkl_dir ./demo/prepared \
    --gpu 0 \
    --batch_size 64 \
    --epochs 50

# 4. Extract features for downstream tasks
bash scripts/3_extract_features.sh \
    --input_dir ./demo/TCGA \
    --gpu 0

# 5. Run downstream evaluation
bash scripts/4_run_downstream.sh \
    --features_dir ./demo/TCGA/features \
    --clinical_dir ./demo/clinical
```

## Demo Data

We provide a small demo dataset to help you get started quickly:

| Cohort | # Images | Cancer Type | Description |
|--------|----------|-------------|-------------|
| BRCA   | 78       | Breast      | Invasive breast carcinoma |
| LUAD   | 78       | Lung        | Lung adenocarcinoma |

### Download

📥 **[Download Demo Data](#)** (coming soon)

### Expected Directory Structure

```
demo/
├── TCGA/
│   ├── BRCA/
│   │   ├── TCGA-3C-AALI-01Z-00-DX1.png
│   │   ├── TCGA-3C-AALI-01Z-00-DX1.png
│   │   ├── ...
│   │   └── segment/                    # Optional: pre-computed masks
│   │       ├── TCGA-3C-AALI-01Z-00-DX1.png
│   │       └── ...
│   └── LUAD/
│       ├── ...
│       └── segment/
└── clinical/
    ├── survival/
    │   └── BRCA.survival.csv
    ├── mutation/
    │   └── BRCA.all
    └── drug/
        └── drug.csv
```

Each image should be a **1000×1000 pixel** ROI patch (region of interest) cropped from a diagnostic whole-slide image, in PNG, JPG, or TIF format.

## Data Preparation

CPSformer operates on **cell-level graphs** extracted from ROI images. The data preparation pipeline converts your images into training-ready PKL files.

### Automatic Nucleus Segmentation (Recommended)

If you only have ROI images (no segmentation masks), CPSformer will **automatically segment nuclei** using our built-in:

- **DeepLabV3** (ResNet50 backbone, trained on IHC histopathology) — **included**
- Followed by **Watershed splitting** for touching nuclei

```
ROI Image (1000×1000)
        │
        ▼
  ┌─────────────┐
  │ DeepLabV3   │──▶ binary mask
  │ (ResNet50)  │
  └─────────────┘
        │
        ▼
    Watershed Split
        │
        ▼
  Cell patches (56×56) + Centroid coordinates
        │
        ▼
    Training PKL file
```

> **Note:** An optional UNet+Attention ensemble model is available upon request for marginally better segmentation quality (adds ~519 MB). The default DeepLabV3 alone provides excellent results for most pathology images.

### One-Click Preparation

```bash
bash scripts/1_prepare_data.sh \
    --input_dir ./my_data \
    --output_dir ./prepared_data \
    --gpu 0
```

This script:
- Scans each cohort subdirectory for images
- Automatically runs segmentation for images without masks
- Extracts cell patches (56×56) and centroid coordinates
- Saves to PKL format

### Using Your Own Segmentation

You can provide pre-computed segmentation masks:

```
my_data/
├── BRCA/
│   ├── image1.png
│   ├── image2.png
│   └── segment/           # Put masks here
│       ├── image1.png     # Binary mask (255=nucleus, 0=background)
│       └── image2.png
```

Run with `--skip_seg` to use existing masks:

```bash
bash scripts/1_prepare_data.sh \
    --input_dir ./my_data \
    --output_dir ./prepared_data \
    --skip_seg
```

### Replacing the Segmentation Model

To use a custom segmentation model, replace the `run_deeplabv3_seg()` function in `nucseg_modules/nucseg_deeplabv3.py`. Your function should return:

```python
def run_deeplabv3_seg(image_paths, work_dir, gpu_id=0):
    """
    Custom nucleus segmentation function.
    
    Args:
        image_paths: List of paths to ROI images
        work_dir: Directory for intermediate files
        gpu_id: GPU device ID
        
    Returns:
        dict: {filename: numpy_mask}  # mask is H×W uint8 (0=background, 255=nucleus)
    """
    # Load and run your custom model here
    ...
```

### PKL File Format

Each PKL file contains a dictionary with the following keys:

| Key | Type | Description |
|-----|------|-------------|
| `x_samplename` | `list[str]` | Patient/sample ID |
| `x_imgname` | `list[str]` | Image filename |
| `x_nucpatch` | `list[np.ndarray]` | Cell patches, each `[N, 56, 56, 3]` uint8 |
| `x_nucpatch_pos` | `list[np.ndarray]` | Cell centroid coordinates, each `[N, 2]` float32 (pixel space [0, 1000)) |
| `x_tumor` | `list[str]` | Tumor type / cohort label |

### Extracted Feature Format

After feature extraction, each cohort produces a CSV file:

```csv
samplename,imgname,feature_0,feature_1,...,feature_1023
TCGA-XX-XXXX,TCGA-XX-XXXX-01Z-00-DX1.png,0.123,-0.456,...,0.789
```

## Training

CPSformer uses **pre-trained weights + fine-tuning**: start from weights pre-trained on 24 TCGA cancer types, then fine-tune on your own dataset.

### One-Click Fine-Tuning

```bash
bash scripts/2_train_finetune.sh \
    --pkl_dir ./prepared_data \
    --gpu 0 \
    --batch_size 64 \
    --epochs 100
```

### Direct Python Usage

```bash
python train_single_cohort.py \
    --pkl_dir ./prepared_data \
    --pretrained_model_path ./checkpoints/best_model.pth \
    --distilled_cell_path ./checkpoints/checkpoints_cellfeature/model.pth \
    --checkpoints_dir ./checkpoints_finetuned \
    --batch_size 64 \
    --epoch_count 100 \
    --lr 5e-5 \
    --gpu_id 0 \
    --gradient_checkpointing \
    --encoder_chunk_size 32000
```

### Key Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--batch_size` | 32 | Batch size per GPU |
| `--accum_steps` | 1 | Gradient accumulation steps (effective = batch × accum) |
| `--max_cells` | 2500 | Maximum cells per image (reduce if OOM) |
| `--lr` | 5e-5 | Learning rate |
| `--epoch_count` | 200 | Number of training epochs |
| `--alpha` | 0.1 | Instance diversity loss weight |
| `--gamma` | 0.1 | Feature diversity loss weight |
| `--delta` | 0.8 | Classification loss weight |
| `--beta` | 0.1 | Supervised contrastive loss weight |
| `--temp` | 0.1 | Contrastive temperature |
| `--gradient_checkpointing` | flag | Enable Transformer gradient checkpointing (saves memory) |
| `--encoder_chunk_size` | 0 | Chunk size for cell encoder (0=disabled; use 32000 for large batches) |

### Loss Functions

CPSformer uses a composite loss:

```
L_total = w_con × L_contrastive + w_div × L_diversity + w_ins × L_instance + w_cls × L_classification + w_supcon × L_SupCon
```

- **L_contrastive** (NTXentLoss): Cross-view contrastive loss between cell-dropout and subgraph-crop augmentations
- **L_diversity**: Covariance off-diagonal penalty for feature uniformity
- **L_instance**: Instance-level diversity via pairwise cosine similarity
- **L_classification**: Cross-entropy for tumor type classification
- **L_SupCon** (SupConLoss): Supervised contrastive loss using class labels

## Inference / Feature Extraction

Extract 1024-dimensional CPS features from your ROI images:

```bash
bash scripts/3_extract_features.sh \
    --input_dir ./my_data \
    --model_path ./checkpoints/best_model.pth \
    --gpu 0
```

This script will:
1. Check each cohort for pre-computed segmentation masks
2. **Automatically run nucleus segmentation** if masks are missing
3. Extract cell patches and run CPSformer to produce 1024-dim features
4. Save CSV files: `{cohort}.cps_feature.csv`

## Downstream Tasks

CPSformer features support multiple downstream clinical tasks:

### Survival Prognosis Prediction

```bash
python DownstreamTask/downstream_survival.py \
    --features_dir ./features \
    --survival_dir ./data/clinical/survival \
    --output_dir ./results_survival
```

**Input:** `features/{cohort}.cps_feature.csv`, `clinical/survival/{cohort}.survival.csv` (samplename, time, status)
**Output:** C-index per cohort via Cox proportional hazards deep neural network

### Gene Mutation Status Prediction

```bash
python DownstreamTask/downstream_mutation_improved.py \
    --features_dir ./features \
    --mutation_dir ./data/clinical/mutation \
    --output_dir ./results_mutation
```

**Input:** `clinical/mutation/{cohort}.all` (gene × patient binary matrix)
**Output:** AUC per gene via multiple strategies (XGBoost, MLP, Top-K pooling)

### Drug Sensitivity Prediction

```bash
python DownstreamTask/downstream_drug_improved.py \
    --features_dir ./features \
    --drug_csv ./data/clinical/drug.csv \
    --output_dir ./results_drug
```

**Input:** `clinical/drug.csv` (patient, drug1_IC50, drug2_IC50, ...)
**Output:** Spearman correlation per drug via SVR/XGBoost/MLP

### TNM Staging Prediction

```bash
python DownstreamTask/downstream_tnm.py \
    --features_dir ./features \
    --clinical_dir ./data/clinical/clinical \
    --output_dir ./results_tnm
```

**Input:** `clinical/{cohort}_tnm.csv` with `t_stage`, `n_stage`, `m_stage` columns
**Output:** Accuracy and AUC per staging task

### Gleason Grading (PRAD)

```bash
python DownstreamTask/downstream_gleason.py \
    --features_dir ./features \
    --clinical_dir ./data/clinical/clinical \
    --output_dir ./results_gleason
```

**Input:** `clinical/{cohort}_gdc_clinical.csv` with `gleason_score` column
**Output:** Accuracy, AUC, and F1 for ISUP Grade Groups 1–5

### One-Click: Run All Tasks

```bash
bash scripts/4_run_downstream.sh \
    --features_dir ./features \
    --clinical_dir ./data/clinical \
    --task all
```

## Advanced: WSI-Level Analysis

For whole-slide image (SVS format) analysis:

### WSI Tumor Classification

```bash
python wsi_mil_classify.py
```

### Survival Risk Heatmap on WSI

```bash
python DownstreamTask/wsi_survival_heatmap.py \
    --svs_path ./WSIs/patient001.svs \
    --cohort BRCA \
    --cps_model ./checkpoints/best_model.pth
```

### Grad-CAM Visualization

```bash
python gradcam_wsi_v2.py
```

> **Note:** WSI analysis requires `openslide-python` (`pip install openslide-python`).

## Pre-trained Models

Download pre-trained model weights before running inference or fine-tuning:

| Model | File | Size | Description |
|-------|------|------|-------------|
| CPSformer (SupCon) | `checkpoints/best_model.pth` | 118 MB | Full model with SupCon pre-training on 24 TCGA cohorts |
| Cell Encoder | `checkpoints/checkpoints_cellfeature/model.pth` | 43 MB | Distilled ResNet-18 cell feature extractor (from UNI2) |
| Nucleus Segmentation | `checkpoints/nucseg_deeplabv3/models/model.pth` | 152 MB | DeepLabV3 (ResNet50) for automatic nucleus segmentation |

📥 **[Download Pre-trained Models](#)** (coming soon)

## Project Structure

```
CPSformer/
├── README.md
├── requirements.txt
├── LICENSE
├── scripts/                    # One-click scripts
│   ├── setup_env.sh
│   ├── 1_prepare_data.sh
│   ├── 2_train_finetune.sh
│   ├── 3_extract_features.sh
│   └── 4_run_downstream.sh
├── train_single_cohort.py     # Main training script
├── prepare_data.py            # Data preparation pipeline
├── extract_cps_features.py    # Feature extraction
├── utils/
│   ├── DataSets.py             # DatasetLoader
│   ├── models.py                # MILCellModelmerge (GAT + Transformer)
│   └── utils.py                # Loss functions (NTXentLoss, SupConLoss, etc.)
├── nucseg_modules/
│   ├── nucseg_pipeline.py     # Segmentation orchestrator
│   ├── nucseg_deeplabv3.py    # DeepLabV3 segmentation (PyTorch)
│   └── nucseg_unet.py         # UNet segmentation (optional)
├── DownstreamTask/
│   └── downstream_*.py              # Downstream task scripts
└── checkpoints/               # Pre-trained model weights
```

## Performance Benchmark

| Task | Metric | CPSformer | ResNet50 | CONCH | UNI2 |
|------|--------|-----------|----------|-------|------|
| WSI Classification | Accuracy | — | — | — | — |
| Survival (C-index) | Mean ± Std | — | — | — | — |
| Mutation (AUC) | Mean | — | — | — | — |
| Drug (SCC) | Mean | — | — | — | — |

> Results from our paper. Detailed benchmarks will be added upon publication.

## Citation

If you use CPSformer in your research, please cite:

```bibtex
@article{cpsformer2024,
  title={CPSformer: Cell Patch Set Transformer for Pathology Image Analysis},
  author={},
  journal={},
  year={2024}
}
```

## License

This project is released under the BSD 3-Clause License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- Foundation model features: [UNI2](https://github.com/mahmoodlab/UNI), [CONCH](https://github.com/MahmoodLab/CONCH), [TITAN](https://github.com/dccxi/TITAN)
- Nucleus segmentation models trained on IHC histopathology datasets
- TCGA project for providing publicly available cancer genomics and pathology data
