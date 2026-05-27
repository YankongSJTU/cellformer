# CPSFormer: Cell Population Structure Representations via Transformer 

**Automatic modeling of cell population structures in histopathology images using hierarchical Transformer architectures**

## 📖 Overview

CPSFormer introduces a novel framework for analyzing dense cellular images (e.g., H&E slides) through Cell Population Structure (CPS) representations. The method combines:

- Single-cell feature extraction with Transformer encoders
- Graph-based spatial reasoning (GAT + Delaunay triangulation)
- Contrastive learning with diversity regularization

**Key capabilities:**  
✔ Tumor classification & prognosis prediction  
✔ Drug sensitivity estimation  
✔ Spatial pattern retrieval across 24 TCGA tumor types  
✔ Cross-scale similarity detection in low visual-similarity cases  

## 🛠 Installation

```bash
git clone https://github.com/YankongSJTU/cellformer.git
cd cellformer
conda env create -f environment.yml
```

## 🛠 Project Structure
```bash
cellformer/
├── Cellformer.py               - Main pipeline for CPS feature prediction
├── models.py                   - Core model architectures
├── CreateDatasets.py           - H&E image preprocessing and dataset construction
├── utils/
│   ├── utils.py                - Useful functions
│   └── Datasets.py             - Datasets modules
│
├── data/                       - Demo data (with download link)
└── checkpoints/                - saved weights
```

## 🚀 Quick Start
1. Data Preparation
```python
python CreatDataset.py --mode test --datadir DATA_PATH --image_dir IMAGE_FILE_PATH --nuc_seg_dir NUCLEI_SEGMENT_PATH --basenamelen LENGTH_FOR_BASENAME_of_IAMGES
```
3. Extract CPS Features
```python
python CPSFormer.py --testdatadir TESTDATA_PATH --gpu_ids GPU_IDs
```

## 📜 Citation
If you use CellFormer in your research, please cite:
```bibtex
@article{cellformer2026,
  title={Mesoscale Cell Population Topology Decouples Histopathological Representation from Visual Domain Shifts},
  author={Yan Kong et al.},
  journal={***},
  year={2026}
}
```
## 🤝 Contributing
We welcome contributions! If you are interested in contributing to:
    The construction of population features
    The improvement of the contrastive learning module
    The enhancement of the GAT (Graph Attention Network) module
    Or any other related areas
Please feel free to contact me.

## 📧 Contact
For questions, contact: kongyan@sjtu.edu.cn


