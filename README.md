# CPSFormer: Cell Population Structure Representations via Transformer 

**Automatic modeling of cell population structures in histopathology images using hierarchical Transformer architectures**

## 📖 Overview

CPSformer is a self-supervised pre-training framework for pathology image analysis. The model extracts cell-level visual features, models spatial topology relationships via Graph Attention Networks (GAT), and aggregates global context through Transformers to generate 1024-dimensional image features. These features can be used for downstream tasks including tumor classification, mutation prediction, drug sensitivity prediction, and survival analysis.

## Key Features

- **Topology-Aware**: Models spatial relationships between cells via GAT with 2D position embeddings
- **Cross-Scale Invariance**: Achieved through random subgraph crop augmentation during training
- **Dual Pooling Mechanism**: Combines mean pooling with sparse attention pooling
- **Flexible Segmentation**: Supports semantic segmentation masks + watershed algorithm for nuclei detection


## Installation

### Dependencies

```bash
# Core dependencies
pip install torch torchvision
pip install torch_geometric
pip install scipy scikit-image scikit-learn
pip install opencv-python Pillow tqdm pandas numpy

# Optional dependencies
pip install umap-learn matplotlib seaborn  # for visualization
```

### Requirements

- Python 3.8+
- PyTorch 1.12+
- CUDA-capable GPU (recommended)

---
## Data & Model Downloads

| Item | Description | Size | Link |
|------|-------------|------|------|
| Demo Data | Sample training data (22 TCGA cohorts, ~10 images each) | ~800 MB | [Google Drive](https://drive.google.com/file/d/1MMHdEp5s6MmZD-H1JQlOisHk5XBa1z_K/view?usp=drive_link) |
| Pretrained Model | CPSformer checkpoint trained on 22 TCGA cancer types | ~50 MB | [Google Drive](https://drive.google.com/file/d/1n9AvL2jznr2npJaTGEBC_xoDm37VQ68M/view?usp=drive_link) |
| Cell Encoder | Distilled ResNet-18 cell encoder (from UNI2) | ~45 MB | [Google Drive](https://drive.google.com/file/d/1SSJfQQPKhvxBGhlL9p3N0Jaava-hRV6N/view?usp=drive_link) |

After downloading, organize files as follows:
```
CPSformer/
├── data/
│   ├── image/                    # Demo tissue images (PNG)
│   ├── segment/                  # Demo segmentation masks (PNG)
│   └── merged_Demo_train_data.pkl  # Demo training data
├── checkpoints/
│   ├── pretrain/
│   │   └── best_model.pth        # Main pretrained model
│   └── cell_distill/
│   │   └── model.pth             # Cell encoder weights
```

---


### Option 1: Extract Features from Your Own Images

Prepare your data with image files and corresponding segmentation masks:

```
data/
├── image/
│   ├── sample1.png
│   ├── sample2.png
│   └── ...
├── segment/
│   ├── sample1.png    # Same filename as image
│   ├── sample2.png
│   └── ...
```

**Segmentation mask requirements:**
- Grayscale PNG format
- Nuclei pixels > 0, background = 0
- Same dimensions as the image

Run feature extraction:

```bash
python src/extract_features.py \
    --image_dir data/image \
    --segment_dir data/segment \
    --checkpoint checkpoints/pretrain/best_model.pth \
    --distilled_path checkpoints/cell_distill/model.pth \
    --output results/features.csv \
    --batch_size 16 \
    --gpu 0
```

**Output format:** CSV with 1026 columns
- Columns `0`-`1023`: 1024-dimensional L2-normalized feature vector
- Column `samplename`: Patient ID (first 12 characters of filename)
- Column `imgname`: Original filename

### Option 2: Extract Features from Pre-processed Data

If you have a pre-processed pickle file:

```bash
python src/extract_features.py \
    --pkl_path data/merged_Demo_train_data.pkl \
    --checkpoint checkpoints/pretrain/best_model.pth \
    --output results/features.csv
```

---

## Training from Scratch

### Step 1: Prepare Training Data

Organize TCGA-style data by cohort:

```
data/
├── dataBLCA/
│   ├── image/
│   └── segment/
├── dataBRCA/
│   ├── image/
│   └── segment/
├── ...
```

Run data preparation:

```bash
python src/create_merged.py \
    --root_dir /path/to/data \
    --save_path data/merged_train_data.pkl \
    --samples_per_type 1000 \
    --max_cells 2500 \
    --num_workers 12
```

**Parameters:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--root_dir` | Required | Root directory containing cohort folders |
| `--save_path` | Required | Output pickle file path |
| `--samples_per_type` | 1000 | Max images per tumor type |
| `--max_cells` | 2500 | Max cells per image |
| `--num_workers` | 12 | Parallel workers |

### Step 2: Train Model

```bash
CUDA_VISIBLE_DEVICES=0,1 python src/train_main.py \
    --merged_pkl data/merged_train_data.pkl \
    --checkpoints_dir checkpoints/pretrain \
    --distilled_cell_path checkpoints/cell_distill/model.pth \
    --epochs 200 \
    --batch_size 32 \
    --lr 5e-5 \
    --dmodel 256 \
    --featuredim 1024
```

**Training parameters:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 100 | Training epochs |
| `--batch_size` | 32 | Batch size per GPU |
| `--lr` | 5e-5 | Learning rate (AdamW) |
| `--dmodel` | 256 | Hidden dimension |
| `--featuredim` | 1024 | Output feature dimension |
| `--max_cells` | 2500 | Max cells per image |
| `--alpha` | 0.1 | Instance loss weight |
| `--gamma` | 0.1 | Diverse loss weight |
| `--delta` | 0.8 | Classification loss weight |
| `--temp` | 0.1 | Contrastive temperature |
| `--crop_min_frac` | 0.3 | Min crop fraction (cross-scale) |
| `--crop_max_frac` | 0.9 | Max crop fraction (cross-scale) |

**Output:**
- `checkpoints/<name>/best_model.pth` — Best model checkpoint
- `checkpoints/<name>/training_log.csv` — Training log

---

## Model Architecture
```
Input: Cell patches [B, N, 3, 56, 56] + Coordinates [B, N, 2]
              │
        ┌─────┴─────┐
        │ ResNet-18 │  512-dim visual features (distilled from UNI2)
        │ (frozen)  │
        └─────┬─────┘
              │ Linear(512→256)
        ┌─────┴─────┐
        │ + 2D Pos  │  Position embedding (raw pixel coords [0,1000])
        └─────┬─────┘
              │
        ┌─────┴─────┐
        │ 2× GAT    │  Graph attention (KNN graph, z-score normalized)
        └─────┬─────┘
              │
        ┌─────┴─────┐
        │ 2× Trans  │  Transformer encoder (global context)
        └─────┬─────┘
              │
        ┌─────┴─────┐
        │ Dual Pool │  Mean pooling + Sparse attention (4 queries)
        │ [1+4]×256 │
        └─────┬─────┘
              │ MLP + L2-Norm
              │
Output: [B, 1024] normalized features + [B, 24] class logits
```

**Key Design Choices:**
- **Coordinate convention**: Raw pixel values [0, 1000] without pre-normalization
- **Graph construction**: Z-score normalized internally for scale-invariant topology
- **Output features**: L2-normalized, suitable for cosine similarity computation

---

## Training Loss

```
Total Loss = (1-α-γ) × L_contrastive + γ × L_diverse + α × L_instance + δ × L_classification
```

| Loss | Weight | Purpose |
|------|--------|---------|
| InfoNCE Contrastive | 0.8 | Learn discriminative global features |
| Diverse Loss | 0.1 | Increase feature diversity |
| Instance Loss | 0.1 | Instance-level discrimination |
| Cross-Entropy | 0.8 | Tumor type classification supervision |

---

## Downstream Performance

### Mutation Prediction (TCGA 22 Cohorts)

| Method | Cohort-Gene Pairs | Mean AUC |
|--------|-------------------|----------|
| Mean Pooling + SVM | 239 | 0.5087 |
| Top-K Pooling + SVM | 234 | 0.5111 |
| Attention-based NN | 239 | **0.5813** |

Top performing pairs: BLCA TP53 (AUC=0.61), LUAD TP53 (AUC=0.60)

### Survival Analysis (Cox Regression)

| Cohort | N Patients | C-index |
|--------|------------|---------|
| BRCA | 284 | **0.668** |
| DLBC | 41 | **0.647** |
| PAAD | 151 | **0.635** |
| UCEC | 144 | **0.624** |

Mean C-index across 22 cohorts: 0.557

### Drug Sensitivity Prediction

| Method | Drugs | Mean SCC |
|--------|-------|----------|
| Mean Pooling + SVR | 427 | 0.167 |
| Top-K Pooling + SVR | 427 | **0.183** |

---

## Supported Tumor Types

The model is trained on 22 TCGA cancer types:

- BLCA (Bladder), BRCA (Breast), CESC (Cervical), COAD (Colon)
- DLBC (Lymphoma), ESCA (Esophageal), GBM (Glioblastoma)
- HNSC (Head & Neck), KIRC/KIRP (Kidney), LGG (Low-grade Glioma)
- LIHC (Liver), LUAD/LUSC (Lung), OV (Ovarian)
- PAAD (Pancreatic), PRAD (Prostate), READ (Rectal)
- STAD (Stomach), THCA (Thyroid), THYM (Thymoma), UCEC (Uterine)

---

## Troubleshooting

**Q: GPU memory insufficient?**
- Reduce `--batch_size` (e.g., 8 or 16)
- Reduce `--max_cells` (e.g., 1000 or 1500)
- Use fewer GPUs

**Q: How to use custom segmentation masks?**
- Place images in `data/image/` and masks in `data/segment/`
- Masks must be grayscale PNG with nuclei pixels > 0, background = 0
- Filename must match the image filename

**Q: Segmentation mask format issues?**
- The script automatically binarizes masks (threshold = 1)
- Adjust `cv2.threshold` in `extract_features.py` if needed

---

## Citation

If you use CPSformer in your research, please cite:

```bibtex
@article{CPSformer2026,
  title={CPSformer: Cell Position & Structure Transformer for Computational Pathology},
  author={...},
  journal={...},
  year={2026}
}
```

---

## License

MIT License

---

## Acknowledgments

- TCGA dataset for training data
- UNI2 foundation model for distilled cell encoder weights
- 
## 📧 Contact
For questions, contact: kongyan@sjtu.edu.cn


