#!/bin/bash
# setup_env.sh — Create conda environment and install dependencies for CPSformer
set -e

ENV_NAME=${1:-cpsformer}
PYTHON_VERSION=${2:-3.10}

echo "=== CPSformer Environment Setup ==="
echo "Environment name: ${ENV_NAME}"
echo "Python version: ${PYTHON_VERSION}"

# Create conda environment
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Conda environment '${ENV_NAME}' already exists. Skipping creation."
else
    echo "Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
    conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
fi

# Activate environment
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

# Install PyTorch (adjust CUDA version as needed)
echo ""
echo "Installing PyTorch..."
echo "Please visit https://pytorch.org/ to select the correct CUDA version for your system."
echo "Default: PyTorch with CUDA 11.8"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric
echo ""
echo "Installing PyTorch Geometric..."
pip install torch-geometric

# Install requirements
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
pip install -r "${PROJECT_ROOT}/requirements.txt"

# Install segmentation dependencies (used by nucseg_modules)
echo ""
echo "Installing segmentation dependencies..."
pip install segmentation-models-pytorch albumentations PyYAML scikit-image

# Verify installation
echo ""
echo "=== Verification ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
import torchvision
print(f'torchvision: {torchvision.__version__}')
import torch_geometric
print(f'torch_geometric: {torch_geometric.__version__}')
import numpy, pandas, sklearn, cv2
print('numpy, pandas, scikit-learn, opencv: OK')
import segmentation_models_pytorch, albumentations, yaml, skimage
print('segmentation_models_pytorch, albumentations, PyYAML, scikit-image: OK')
"

echo ""
echo "=== Setup complete! ==="
echo "Activate with: conda activate ${ENV_NAME}"
