#!/usr/bin/env python3
"""
WSI Survival Risk Heatmap Pipeline
Full pipeline: SVS → patch extraction → nuclear segmentation → CPS features → survival heatmap

Author: Yan Kong
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import pickle
import argparse
import random
from tqdm import tqdm
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import openslide
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
from models import MILCellModelmerge


def parse_args():
    parser = argparse.ArgumentParser(description="WSI Survival Risk Heatmap")
    parser.add_argument('--svs_path', type=str,
                        default='/data1/TumorGroup/DATA/public_database/TCGA/slide/BRCA/1a62d692-bc08-49f9-ad59-6fddf3bbcb6d/TCGA-A7-A4SF-01Z-00-DX1.CDCFD4BC-4363-4CF2-95F5-4922E04C3B9D.svs')
    parser.add_argument('--cohort', type=str, default='BRCA')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--cps_model', type=str,
                        default=os.path.join(PROJECT_ROOT, 'checkpoints_supcon', 'best_model.pth'))
    parser.add_argument('--cell_encoder', type=str,
                        default=os.path.join(PROJECT_ROOT, 'checkpoints_cell', 'model.pth'))
    parser.add_argument('--seg_model', type=str,
                        default=os.path.join(PROJECT_ROOT, 'checkpoints', 'nucseg_deeplabv3', 'models', 'model.pth'))
    parser.add_argument('--features_dir', type=str,
                        default=os.path.join(PROJECT_ROOT, 'features'))
    parser.add_argument('--survival_dir', type=str,
                        default=os.path.join(PROJECT_ROOT, 'clinical', 'survival'))
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(PROJECT_ROOT, 'figures_wsi_heatmap'))
    parser.add_argument('--patch_size', type=int, default=1000,
                        help='Patch size at 40X (1000x1000)')
    parser.add_argument('--cell_patch_size', type=int, default=56,
                        help='Cell patch size for CPS features (56x56)')
    parser.add_argument('--max_cells_per_patch', type=int, default=150,
                        help='Max cells to extract per patch (for memory)')
    parser.add_argument('--min_cells_per_patch', type=int, default=15,
                        help='Min cells needed for valid patch')
    parser.add_argument('--tissue_threshold', type=float, default=0.05,
                        help='Min tissue ratio to keep a patch')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for segmentation')
    return parser.parse_args()


# ========================================
# Models
# ========================================

def load_nuclear_segmentation_model(opt):
    """Load DeepLabV3 for nuclear segmentation"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torch.hub.load('pytorch/vision:v0.11.2', 'deeplabv3_resnet50', pretrained=False)
    model = nn.DataParallel(model, device_ids=[0])
    model.load_state_dict(torch.load(opt.seg_model, map_location=device), strict=False)
    model.eval()
    model.to(device)

    return model, device


def load_cps_model(opt):
    """Load CPSformer model for feature extraction"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MILCellModelmerge(
        num_classes=24,
        d_model=256,
        output_dim=1024,
        distilled_path=opt.cell_encoder
    ).to(device)

    if os.path.exists(opt.cps_model):
        checkpoint = torch.load(opt.cps_model, map_location=device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            if name.startswith('cell_encoder.'):
                continue
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded CPS model from {opt.cps_model}")

    model.eval()
    return model, device


def load_survival_model(opt):
    """Load or train survival prediction model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load features and survival data for training
    feature_files = [
        os.path.join(opt.features_dir, f'{opt.cohort}.cps_feature.csv'),
        os.path.join(opt.features_dir, f'{opt.cohort}1.cps_feature.csv'),
        os.path.join(opt.features_dir, f'{opt.cohort}2.cps_feature.csv'),
    ]

    df = None
    for fpath in feature_files:
        if os.path.exists(fpath):
            _df = pd.read_csv(fpath)
            df = _df if df is None else pd.concat([df, _df], ignore_index=True)

    if df is None:
        print(f"No features found for {opt.cohort}")
        return None, None, device

    feature_cols = [c for c in df.columns if c not in ('samplename', 'imgname')]

    # Aggregate per patient (mean pooling)
    patient_features = df.groupby('samplename')[feature_cols].mean().reset_index()

    # Load survival data
    surv_path = os.path.join(opt.survival_dir, f'{opt.cohort}.survival.csv')
    if not os.path.exists(surv_path):
        print(f"No survival data for {opt.cohort}")
        return None, None, device

    surv_df = pd.read_csv(surv_path, sep='\t')

    # Merge
    common = set(patient_features['samplename']) & set(surv_df['samplename'])
    if len(common) < 30:
        print(f"Too few common patients: {len(common)}")
        return None, None, device

    merged = pd.merge(
        patient_features[patient_features['samplename'].isin(common)],
        surv_df[surv_df['samplename'].isin(common)],
        on='samplename'
    )

    X = merged[feature_cols].values
    times = merged['time'].values.astype(np.float32)
    events = merged['status'].values.astype(np.float32)

    print(f"Training survival model on {len(X)} patients")

    # Train survival model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = nn.Sequential(
        nn.Linear(X.shape[1], 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 1)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    best_ci = 0
    best_state = None

    for epoch in range(100):
        model.train()
        X_t = torch.FloatTensor(X_scaled).to(device)
        t_t = torch.FloatTensor(times).to(device)
        e_t = torch.FloatTensor(events).to(device)

        risk = model(X_t).squeeze()

        # Cox loss
        n = len(t_t)
        R_mat = (t_t.unsqueeze(1) <= t_t.unsqueeze(0)).float()
        exp_risk = torch.exp(risk)
        log_sum = torch.log(torch.sum(exp_risk * R_mat, dim=1) + 1e-8)
        loss = -torch.mean((risk - log_sum) * e_t)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # C-index
        model.eval()
        with torch.no_grad():
            risk_np = model(torch.FloatTensor(X_scaled).to(device)).cpu().numpy().flatten()
            ci = c_index(risk_np, times, events)

        if ci > best_ci:
            best_ci = ci
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Load best state
    if best_state:
        model.load_state_dict(best_state)

    print(f"Survival model C-index: {best_ci:.4f}")
    model.eval()

    return model, scaler, device


def c_index(risk, time, event):
    """Compute concordance index"""
    n = len(time)
    if n < 2:
        return 0.5
    risk = risk.flatten()
    time = time.flatten()
    event = event.flatten()
    concordant = 0
    permissible = 0
    for i in range(n):
        if event[i] == 0:
            continue
        for j in range(n):
            if time[j] < time[i]:
                permissible += 1
                if risk[j] > risk[i]:
                    concordant += 1
                elif risk[j] == risk[i]:
                    concordant += 0.5
    return concordant / max(permissible, 1)


# ========================================
# Patch extraction
# ========================================

def detect_tissue(slide, threshold=200):
    """Detect tissue regions from low-res thumbnail"""
    # Use the lowest resolution level available
    level = slide.level_count - 1
    thumb = slide.read_region((0, 0), level, slide.level_dimensions[level])
    thumb = np.array(thumb)

    # Convert to grayscale
    gray = cv2.cvtColor(thumb, cv2.COLOR_RGBA2GRAY)

    # Threshold - tissue is darker
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get bounding boxes
    tissue_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 20 and h > 20:  # Filter small noise
            tissue_boxes.append((x, y, w, h))

    return tissue_boxes, thumb, binary


def extract_patches_with_coords(slide, tissue_boxes, level_downsample, patch_size=1000):
    """
    Extract 1000x1000 patches at 40X from tissue regions.

    Returns:
        patches: list of image arrays
        coords: list of (x, y) coordinates at level 0
    """
    patches = []
    coords = []

    # Calculate scale factor from thumbnail level to level 0
    thumb_level = slide.level_count - 1
    scale = slide.level_downsamples[thumb_level]

    for box in tissue_boxes:
        tx, ty, tw, th = box

        # Convert thumbnail coords to level 0 coords
        x0 = int(tx * scale)
        y0 = int(ty * scale)

        # Calculate dimensions at level 0
        w0 = int(tw * scale)
        h0 = int(th * scale)

        # Extract patches within this tissue region
        for x in range(x0, x0 + w0, patch_size):
            for y in range(y0, y0 + h0, patch_size):
                # Ensure we don't exceed slide boundaries
                w, h = slide.level_dimensions[0]
                if x + patch_size > w or y + patch_size > h:
                    continue

                patch = slide.read_region((x, y), 0, (patch_size, patch_size))
                patch = np.array(patch)
                patch = cv2.cvtColor(patch, cv2.COLOR_RGBA2RGB)

                patches.append(patch)
                coords.append((x, y))

    return patches, coords


def filter_patches_by_tissue(patches, coords, threshold=0.05):
    """Filter patches to keep only those with sufficient tissue"""
    valid_patches = []
    valid_coords = []

    for patch, coord in zip(patches, coords):
        # Convert to grayscale
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)

        # Check tissue ratio (dark pixels = tissue)
        tissue_ratio = np.sum(gray < 200) / gray.size

        if tissue_ratio > threshold:
            valid_patches.append(patch)
            valid_coords.append(coord)

    return valid_patches, valid_coords


# ========================================
# Nuclear segmentation
# ========================================

def segment_nuclei_batch(seg_model, patches, device, batch_size=8):
    """Run nuclear segmentation on batch of patches"""
    masks = []

    seg_model.eval()

    # Process in batches
    for i in tqdm(range(0, len(patches), batch_size), desc="Segmentation"):
        batch = patches[i:i+batch_size]

        # Prepare inputs
        batch_tensors = []
        for patch in batch:
            # Normalize
            patch_norm = patch.astype(np.float32) / 255.0
            patch_tensor = torch.from_numpy(patch_norm).permute(2, 0, 1).unsqueeze(0)
            batch_tensors.append(patch_tensor)

        batch_tensor = torch.cat(batch_tensors, dim=0).to(device)

        # Inference
        with torch.no_grad():
            output = seg_model(batch_tensor)
            # DeepLabV3 returns dict with 'out'
            if isinstance(output, dict):
                output = output['out']

            # Get mask for nuclei (class 0)
            masks_batch = torch.sigmoid(output[:, 0:1, :, :]).cpu().numpy()

        # Threshold
        for j, mask in enumerate(masks_batch):
            mask_uint8 = (mask[0] * 255).astype(np.uint8)
            _, binary = cv2.threshold(mask_uint8, 220, 255, cv2.THRESH_BINARY)
            masks.append(binary)

    return masks


# ========================================
# Cell extraction
# ========================================

def extract_cells_from_mask(img_rgb, seg_mask, patch_size=56, max_cells=300):
    """
    Extract cell patches from segmentation mask using watershed.

    Returns:
        patches: (N, 3, 56, 56) normalized cell patches
        positions: (N, 2) cell centroids in (x, y) format
        n_cells: number of cells
    """
    # Distance transform
    distance = ndi.distance_transform_edt(seg_mask)

    # Find local maxima (nuclei centers)
    coords = peak_local_max(distance, min_distance=11, labels=seg_mask)

    if len(coords) == 0:
        return None, None, 0

    # Limit cells
    if len(coords) > max_cells:
        idx = np.random.choice(len(coords), max_cells, replace=False)
        coords = coords[idx]

    n_cells = len(coords)
    cell_patches = []
    positions = []
    radius = patch_size // 2
    h, w = img_rgb.shape[:2]

    for y, x in coords:  # peak_local_max returns (y, x)
        # Crop with boundary handling
        y1, y2 = max(0, int(y - radius)), min(h, int(y + radius))
        x1, x2 = max(0, int(x - radius)), min(w, int(x + radius))

        crop = img_rgb[y1:y2, x1:x2]

        # Padding if needed
        if crop.shape[0] < patch_size or crop.shape[1] < patch_size:
            pad_h = patch_size - crop.shape[0]
            pad_w = patch_size - crop.shape[1]
            crop = cv2.copyMakeBorder(crop, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

        # Resize to exact size
        if crop.shape[0] != patch_size or crop.shape[1] != patch_size:
            crop = cv2.resize(crop, (patch_size, patch_size))

        cell_patches.append(crop)
        positions.append([x, y])  # Store as (x, y) for CPSformer

    # Convert to tensor format
    cell_patches = np.array(cell_patches, dtype=np.float32)
    cell_patches = cell_patches.transpose(0, 3, 1, 2) / 255.0  # (N, 3, H, W)
    positions = np.array(positions, dtype=np.float32)

    return cell_patches, positions, n_cells


# ========================================
# CPS Feature extraction
# ========================================

def extract_cps_features(cps_model, patches, masks, device, max_cells=300, min_cells=15):
    """
    Extract CPS features for each patch.

    Returns:
        features: (N_patches, 1024) features
        valid_idx: indices of valid patches (with enough cells)
    """
    import faulthandler
    faulthandler.enable()

    features = []
    valid_idx = []

    cps_model.eval()

    for i, (patch, mask) in tqdm(enumerate(zip(patches, masks)), desc="CPS Features", total=len(patches)):
        try:
            # Clear cache periodically
            if i > 0 and i % 30 == 0:
                torch.cuda.empty_cache()
                import gc
                gc.collect()

            cell_patches, positions, n_cells = extract_cells_from_mask(
                patch, mask, patch_size=56, max_cells=max_cells
            )

            if n_cells < min_cells:
                continue

            # Limit to reasonable number of cells to prevent memory overflow
            max_process = 150
            if n_cells > max_process:
                idx = np.random.choice(n_cells, max_process, replace=False)
                cell_patches = cell_patches[idx]
                positions = positions[idx]
                n_cells = max_process

            # Normalize positions to [0, 1000) range for CPSformer
            # Positions are in pixel coords within the 1000x1000 patch
            # Clamp to valid range
            positions = np.clip(positions, 0, 999)

            # Ensure proper data types
            cell_patches = np.ascontiguousarray(cell_patches, dtype=np.float32)
            positions = np.ascontiguousarray(positions, dtype=np.float32)

            # Prepare tensors
            cell_tensor = torch.from_numpy(cell_patches).unsqueeze(0).to(device)  # (1, N, 3, H, W)
            pos_tensor = torch.from_numpy(positions).unsqueeze(0).to(device)      # (1, N, 2)
            mask_tensor = torch.ones(1, n_cells, dtype=torch.float32).to(device)   # (1, N)

            # Inference
            with torch.no_grad():
                feat, _, _ = cps_model(cell_tensor, pos_tensor, mask_tensor)

            features.append(feat.cpu().numpy().flatten())
            valid_idx.append(i)

            # Move tensors back to CPU to free GPU memory
            del cell_tensor, pos_tensor, mask_tensor, feat

        except RuntimeError as e:
            print(f"RuntimeError on patch {i} (n_cells={n_cells if 'n_cells' in dir() else 'unknown'}): {e}")
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"Error processing patch {i}: {type(e).__name__}: {e}")
            continue

    if len(features) == 0:
        return None, []

    return np.array(features), valid_idx


# ========================================
# Visualization
# ========================================

def create_heatmap(coords, values, wsi_dims, patch_size=1000, output_path=None):
    """
    Create risk heatmap from patch coordinates and values.
    """
    w, h = wsi_dims

    # Create heatmap array
    heatmap = np.zeros((h // patch_size + 1, w // patch_size + 1))
    heatmap[:] = np.nan  # Fill with NaN for empty regions

    # Fill in values
    for (x, y), val in zip(coords, values):
        i = y // patch_size
        j = x // patch_size
        heatmap[i, j] = val

    return heatmap


def plot_wsi_heatmap(slide, coords, risk_scores, patch_size=1000, patient_id='', output_path=None):
    """
    Plot WSI thumbnail with risk heatmap overlay.
    """
    # Get thumbnail
    thumb_level = min(2, slide.level_count - 1)
    thumb_dims = slide.level_dimensions[thumb_level]
    # Ensure dimensions don't exceed reasonable limits
    if thumb_dims[0] > 4000 or thumb_dims[1] > 4000:
        thumb_level = slide.level_count - 1
    thumb = slide.read_region((0, 0), thumb_level, slide.level_dimensions[thumb_level])
    thumb = np.array(thumb)
    thumb = cv2.cvtColor(thumb, cv2.COLOR_RGBA2RGB)

    # Get dimensions
    w0, h0 = slide.level_dimensions[0]
    thumb_w, thumb_h = slide.level_dimensions[thumb_level]

    # Scale factors
    scale_x = thumb_w / w0
    scale_y = thumb_h / h0

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Original thumbnail
    axes[0].imshow(thumb)
    axes[0].set_title('WSI Thumbnail', fontsize=14)
    axes[0].axis('off')

    # 2. Risk heatmap
    # Normalize risk scores to 0-1
    risk_min, risk_max = risk_scores.min(), risk_scores.max()
    if risk_max > risk_min:
        risk_norm = (risk_scores - risk_min) / (risk_max - risk_min)
    else:
        risk_norm = np.zeros_like(risk_scores)

    # Create heatmap overlay
    heatmap_img = np.zeros((thumb_h, thumb_w))
    heatmap_count = np.zeros((thumb_h, thumb_w))

    patch_size_thumb = int(patch_size * scale_x)

    for (x, y), rn in zip(coords, risk_norm):
        xt = int(x * scale_x)
        yt = int(y * scale_y)

        # Add to heatmap
        y_end = min(yt + patch_size_thumb, thumb_h)
        x_end = min(xt + patch_size_thumb, thumb_w)
        heatmap_img[yt:y_end, xt:x_end] += rn
        heatmap_count[yt:y_end, xt:x_end] += 1

    # Average overlapping regions
    heatmap_avg = np.where(heatmap_count > 0, heatmap_img / heatmap_count, 0)

    # Create colormap (blue=low risk, red=high risk)
    cmap = plt.cm.RdYlBu_r

    axes[1].imshow(thumb)
    im = axes[1].imshow(heatmap_avg, cmap=cmap, alpha=0.5, vmin=0, vmax=1)
    axes[1].set_title('Risk Heatmap', fontsize=14)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label='Risk (normalized)')

    # 3. High/Low risk patches comparison
    # Sort by risk
    sorted_idx = np.argsort(risk_scores)
    high_risk_idx = sorted_idx[-5:]  # Top 5 high risk
    low_risk_idx = sorted_idx[:5]    # Top 5 low risk

    # Create mini-thumbnails for high and low risk patches
    n_high = len(high_risk_idx)
    n_low = len(low_risk_idx)

    # Draw markers on thumbnail
    ax = axes[2]
    ax.imshow(thumb)

    # Mark high risk patches (red squares)
    for idx in high_risk_idx:
        x, y = coords[idx]
        xt = int(x * scale_x)
        yt = int(y * scale_y)
        rect = plt.Rectangle((xt, yt), patch_size_thumb, patch_size_thumb,
                              fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

    # Mark low risk patches (blue squares)
    for idx in low_risk_idx:
        x, y = coords[idx]
        xt = int(x * scale_x)
        yt = int(y * scale_y)
        rect = plt.Rectangle((xt, yt), patch_size_thumb, patch_size_thumb,
                              fill=False, edgecolor='blue', linewidth=2)
        ax.add_patch(rect)

    ax.set_title(f'Red: High Risk (n={n_high}) | Blue: Low Risk (n={n_low})', fontsize=14)
    ax.axis('off')

    plt.suptitle(f'Survival Risk Heatmap - {patient_id}', fontsize=16, y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.savefig(output_path.replace('.png', '.tif'), dpi=150, bbox_inches='tight', format='tiff')
        print(f"Saved to {output_path}")

    plt.close()

    return heatmap_avg


def plot_patch_examples(patches, coords, risk_scores, masks, output_dir, n_examples=10):
    """
    Show example patches with their risk scores and segmentation masks.
    """
    # Sort by risk
    sorted_idx = np.argsort(risk_scores)

    # Select high, medium, low risk patches
    n = len(sorted_idx)
    indices = [
        sorted_idx[-1], sorted_idx[-2], sorted_idx[-3],  # High risk
        sorted_idx[n//2], sorted_idx[n//2+1],            # Medium risk
        sorted_idx[0], sorted_idx[1], sorted_idx[2],     # Low risk
    ]

    fig, axes = plt.subplots(len(indices), 3, figsize=(12, 4*len(indices)))

    for row, idx in enumerate(indices):
        patch = patches[idx]
        mask = masks[idx]
        risk = risk_scores[idx]
        x, y = coords[idx]

        # Original patch
        axes[row, 0].imshow(patch)
        axes[row, 0].set_title(f'Patch ({x},{y})')
        axes[row, 0].axis('off')

        # Segmentation mask
        axes[row, 1].imshow(mask, cmap='gray')
        axes[row, 1].set_title(f'Nuclei Seg ({np.sum(mask>0)} cells)')
        axes[row, 1].axis('off')

        # Overlay
        overlay = patch.copy()
        overlay[mask > 0] = [255, 0, 0]  # Red overlay on nuclei
        blended = cv2.addWeighted(patch, 0.7, overlay, 0.3, 0)
        axes[row, 2].imshow(blended)
        axes[row, 2].set_title(f'Risk: {risk:.3f}')
        axes[row, 2].axis('off')

    plt.suptitle('Patch Examples with Risk Scores', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'patch_examples.png'), dpi=150)
    plt.close()


# ========================================
# Main pipeline
# ========================================

def main():
    opt = parse_args()
    os.makedirs(opt.output_dir, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

    random.seed(42)

    print(f"\n{'='*60}")
    print("WSI Survival Risk Heatmap Pipeline")
    print(f"{'='*60}")

    # Extract patient ID from filename
    patient_id = os.path.basename(opt.svs_path).split('.')[0][:12]
    print(f"Patient ID: {patient_id}")
    print(f"SVS path: {opt.svs_path}")

    # Step 1: Open SVS
    print("\n[1] Opening SVS file...")
    slide = openslide.open_slide(opt.svs_path)
    print(f"  Level 0 dimensions: {slide.level_dimensions[0]}")
    print(f"  Levels: {slide.level_count}")

    # Step 2: Detect tissue
    print("\n[2] Detecting tissue regions...")
    tissue_boxes, thumb, binary = detect_tissue(slide)
    print(f"  Found {len(tissue_boxes)} tissue regions")

    # Save tissue detection preview
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(thumb)
    axes[0].set_title('Thumbnail')
    axes[0].axis('off')
    axes[1].imshow(binary, cmap='gray')
    axes[1].set_title('Tissue Detection')
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(opt.output_dir, 'tissue_detection.png'), dpi=100)
    plt.close()

    # Step 3: Extract patches
    print("\n[3] Extracting patches at 40X...")
    patches, coords = extract_patches_with_coords(
        slide, tissue_boxes, slide.level_downsamples, patch_size=opt.patch_size
    )
    print(f"  Extracted {len(patches)} initial patches")

    # Filter by tissue content
    patches, coords = filter_patches_by_tissue(patches, coords, threshold=opt.tissue_threshold)
    print(f"  After tissue filtering: {len(patches)} patches")

    if len(patches) == 0:
        print("No valid patches found!")
        return

    # Step 4: Load models
    print("\n[4] Loading models...")
    seg_model, seg_device = load_nuclear_segmentation_model(opt)
    cps_model, cps_device = load_cps_model(opt)
    surv_model, surv_scaler, surv_device = load_survival_model(opt)

    if surv_model is None:
        print("Failed to load/train survival model!")
        return

    # Step 5: Nuclear segmentation
    print("\n[5] Running nuclear segmentation...")
    masks = segment_nuclei_batch(seg_model, patches, seg_device, batch_size=opt.batch_size)

    # Free segmentation model to save GPU memory for CPS features
    print("  Freeing segmentation model...")
    del seg_model
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    # Step 6: CPS feature extraction
    print("\n[6] Extracting CPS features...")
    features, valid_idx = extract_cps_features(
        cps_model, patches, masks, cps_device,
        max_cells=opt.max_cells_per_patch,
        min_cells=opt.min_cells_per_patch
    )
    print(f"  Valid patches with enough cells: {len(valid_idx)}")

    if len(valid_idx) == 0:
        print("No valid patches after cell extraction!")
        return

    # Filter to valid patches
    valid_coords = [coords[i] for i in valid_idx]
    valid_patches = [patches[i] for i in valid_idx]
    valid_masks = [masks[i] for i in valid_idx]

    # Step 7: Predict survival risk for each patch
    print("\n[7] Predicting survival risk...")
    surv_model.eval()

    # Normalize features using survival scaler
    features_scaled = surv_scaler.transform(features)

    with torch.no_grad():
        features_tensor = torch.FloatTensor(features_scaled).to(surv_device)
        risk_scores = surv_model(features_tensor).cpu().numpy().flatten()

    print(f"  Risk scores: min={risk_scores.min():.3f}, max={risk_scores.max():.3f}")
    print(f"  Mean risk: {risk_scores.mean():.3f}")

    # Step 8: Visualize
    print("\n[8] Generating heatmap visualization...")

    # Main heatmap
    plot_wsi_heatmap(
        slide, valid_coords, risk_scores,
        patch_size=opt.patch_size,
        patient_id=patient_id,
        output_path=os.path.join(opt.output_dir, f'{patient_id}_risk_heatmap.png')
    )

    # Patch examples
    plot_patch_examples(
        valid_patches, valid_coords, risk_scores, valid_masks,
        opt.output_dir, n_examples=10
    )

    # Step 9: Save data
    print("\n[9] Saving results...")

    # Save patch data
    results_df = pd.DataFrame({
        'patch_x': [c[0] for c in valid_coords],
        'patch_y': [c[1] for c in valid_coords],
        'risk_score': risk_scores,
        'n_cells': [np.sum(m > 0) for m in valid_masks]
    })
    results_df.to_csv(os.path.join(opt.output_dir, f'{patient_id}_patch_risks.csv'), index=False)

    # Save features
    feature_df = pd.DataFrame(features)
    feature_df['patch_x'] = [c[0] for c in valid_coords]
    feature_df['patch_y'] = [c[1] for c in valid_coords]
    feature_df.to_csv(os.path.join(opt.output_dir, f'{patient_id}_patch_features.csv'), index=False)

    print(f"\n{'='*60}")
    print("Pipeline completed!")
    print(f"Output directory: {opt.output_dir}")
    print(f"{'='*60}")

    slide.close()


if __name__ == "__main__":
    main()