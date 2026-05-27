#!/usr/bin/env python3
"""Extract 1024-dim features from images using trained CPSformer v2 model.

Pipeline:
  1. Read images (data/image/) + semantic segmentation masks (data/segment/)
  2. Apply watershed to separate touching nuclei → centroids
  3. Extract 56×56 cell patches around each centroid
  4. Forward through trained CPSformer model
  5. Save features to CSV (1024 columns + samplename + imgname)

Usage:
  python extract_features.py \
      --image_dir ../data/image \
      --segment_dir ../data/segment \
      --checkpoint ../checkpoints/pretrain/best_model.pth \
      --output ../results/features.csv

  # If you already have a merged pkl file (skip watershed):
  python extract_features.py \
      --pkl_path ../data/merged_data.pkl \
      --checkpoint ../checkpoints/pretrain/best_model.pth \
      --output ../results/features.csv
"""
import argparse
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from torch.utils.data import DataLoader
from tqdm import tqdm


from utils.models import MILCellModelmerge
from utils.DataSets import DatasetLoaderV2
from utils.utils import custom_collate_fn

PATCH_SIZE = 56
CROP_RADIUS = 40


# ── Watershed nuclei segmentation ──────────────────────────────────────────
def watershed_nuclei(binary_mask, min_distance=7):
    """Separate touching nuclei via watershed on distance transform."""
    mask = (binary_mask > 0).astype(np.uint8)
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=np.int32), np.zeros((0, 2), dtype=np.float32)

    distance = ndi.distance_transform_edt(mask)
    coords = peak_local_max(distance, min_distance=min_distance, labels=mask)
    local_max_mask = np.zeros(distance.shape, dtype=bool)
    local_max_mask[tuple(coords.T)] = True
    markers, _ = ndi.label(local_max_mask)
    labels = watershed(-distance, markers, mask=mask)

    max_label = labels.max()
    centroids = []
    for i in range(1, max_label + 1):
        ys, xs = np.where(labels == i)
        if len(xs) == 0:
            continue
        centroids.append([np.mean(xs), np.mean(ys)])

    if not centroids:
        return labels, np.zeros((0, 2), dtype=np.float32)
    return labels, np.array(centroids, dtype=np.float32)


# ── Patch extraction for a single image ────────────────────────────────────
def extract_patches_from_image(img_path, seg_path, max_cells=2500):
    """Return patches [N,56,56,3], coords [N,2] for one image."""
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None, None, None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    seg_gray = cv2.imread(seg_path, 0)
    if seg_gray is None:
        return None, None, None
    _, binary_mask = cv2.threshold(seg_gray, 1, 255, cv2.THRESH_BINARY)

    _, centroids = watershed_nuclei(binary_mask, min_distance=7)
    if len(centroids) == 0:
        return None, None, None

    if len(centroids) > max_cells:
        idx = np.random.choice(len(centroids), max_cells, replace=False)
        centroids = centroids[idx]

    h, w = img_rgb.shape[:2]
    patches, coords = [], []
    for cx, cy in centroids:
        y1 = max(0, int(cy - CROP_RADIUS))
        y2 = min(h, int(cy + CROP_RADIUS))
        x1 = max(0, int(cx - CROP_RADIUS))
        x2 = min(w, int(cx + CROP_RADIUS))

        crop = img_rgb[y1:y2, x1:x2]
        if crop.shape[0] < CROP_RADIUS * 2 or crop.shape[1] < CROP_RADIUS * 2:
            crop = cv2.copyMakeBorder(
                crop,
                max(0, CROP_RADIUS - int(cy)),
                max(0, CROP_RADIUS - (h - int(cy))),
                max(0, CROP_RADIUS - int(cx)),
                max(0, CROP_RADIUS - (w - int(cx))),
                cv2.BORDER_REFLECT)
        if crop.shape[0] != PATCH_SIZE or crop.shape[1] != PATCH_SIZE:
            crop = cv2.resize(crop, (PATCH_SIZE, PATCH_SIZE),
                              interpolation=cv2.INTER_CUBIC)

        patches.append(crop)
        coords.append([cx, cy])

    img_name = os.path.basename(img_path)
    sample_name = img_name[:12]  # TCGA patient ID (first 12 chars)
    return (np.array(patches, dtype=np.uint8),
            np.array(coords, dtype=np.float32),
            {'samplename': sample_name, 'imgname': img_name})


# ── Build dataset from image/segment directories ──────────────────────────
def build_dataset_from_dirs(image_dir, segment_dir, max_cells=2500, batch_size=16):
    """Build DataLoader from raw image + segment pairs."""
    img_files = sorted([f for f in os.listdir(image_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    all_patches, all_pos, all_names, all_labels = [], [], [], []
    skipped = 0

    for fname in tqdm(img_files, desc="Extracting patches"):
        base = os.path.splitext(fname)[0]
        seg_path = os.path.join(segment_dir, base + '.png')
        if not os.path.exists(seg_path):
            seg_path = os.path.join(segment_dir, base + '.jpg')
        if not os.path.exists(seg_path):
            skipped += 1
            continue

        img_path = os.path.join(image_dir, fname)
        patches, coords, meta = extract_patches_from_image(img_path, seg_path, max_cells)
        if patches is None:
            skipped += 1
            continue

        all_patches.append(patches)
        all_pos.append(coords)
        all_names.append(meta['imgname'])
        all_labels.append('Unknown')

    print(f"Processed {len(all_patches)}/{len(img_files)} images "
          f"({skipped} skipped, no matching segment or no cells detected)")

    if not all_patches:
        raise RuntimeError("No valid images found! Check image_dir and segment_dir.")

    data = {
        'x_nucpatch': all_patches,
        'x_nucpatch_pos': all_pos,
        'x_imgname': all_names,
        'x_tumor': all_labels,
    }
    dataset = DatasetLoaderV2(data, is_train=False, max_cells=max_cells)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=custom_collate_fn, num_workers=2,
                        pin_memory=True)
    return loader, all_names


# ── Build dataset from existing pkl ───────────────────────────────────────
def build_dataset_from_pkl(pkl_path, max_cells=2500, batch_size=16):
    """Build DataLoader from an existing merged pkl file."""
    import pickle
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded pkl: {len(data['x_imgname'])} images from {pkl_path}")

    dataset = DatasetLoaderV2(data, is_train=False, max_cells=max_cells)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=custom_collate_fn, num_workers=4,
                        pin_memory=True)
    return loader, data['x_imgname']


# ── Feature extraction loop ───────────────────────────────────────────────
@torch.no_grad()
def extract_features(model, loader, device):
    """Forward all samples through model, collect 1024-dim features."""
    model.eval()
    all_features = []

    for x_patches, x_masks, x_names, labels, pos in tqdm(loader, desc="Extracting features"):
        x_patches = x_patches.to(device)
        x_masks = x_masks.to(device)
        pos = pos.to(device)

        feat, _, _ = model(x_patches, pos, x_masks)
        all_features.append(feat.cpu().numpy())

    return np.concatenate(all_features, axis=0)


# ── Load model ────────────────────────────────────────────────────────────
def load_model(checkpoint_path, distilled_path, device,
               num_classes=24, d_model=256, output_dim=1024):
    model = MILCellModelmerge(
        num_classes=num_classes, d_model=d_model, output_dim=output_dim,
        distilled_path=distilled_path
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
    new_sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(new_sd, strict=False)
    model.eval()
    return model


# ── Main ──────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="CPSformer v2 Feature Extraction")

    # Input source (one required)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--pkl_path', type=str, default=None, help='Path to existing merged pkl file')
    group.add_argument('--image_dir', type=str, default=None, help='Directory of raw tissue images')
    parser.add_argument('--segment_dir', type=str, default=None, help='Directory of segmentation masks (required with --image_dir)')
    parser.add_argument('--checkpoint', type=str, default='../checkpoints/pretrain/best_model.pth', help='Path to trained model checkpoint')
    parser.add_argument('--distilled_path', type=str, default='../checkpoints/cell_distill/model.pth', help='Path to distilled cell encoder weights')
    parser.add_argument('--output', type=str, default='results/extracted_features.csv', help='Output CSV path')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_cells', type=int, default=2500, help='Max cells per image (prevent OOM)')
    parser.add_argument('--gpu', type=str, default='0', help='GPU id')
    return parser.parse_args()


def main():
    opt = parse_args()

    if opt.image_dir and not opt.segment_dir:
        parser.error("--segment_dir is required when using --image_dir")

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    print(f"Loading model from {opt.checkpoint}...")
    model = load_model(opt.checkpoint, opt.distilled_path, device)

    # Build dataset
    if opt.pkl_path:
        loader, img_names = build_dataset_from_pkl(
            opt.pkl_path, opt.max_cells, opt.batch_size)
    else:
        loader, img_names = build_dataset_from_dirs(
            opt.image_dir, opt.segment_dir, opt.max_cells, opt.batch_size)

    # Extract features
    features = extract_features(model, loader, device)
    print(f"Features shape: {features.shape}")

    # Build DataFrame
    df = pd.DataFrame(features, columns=[str(i) for i in range(features.shape[1])])
    df['samplename'] = [n[:12] for n in img_names]
    df['imgname'] = img_names

    # Save
    os.makedirs(os.path.dirname(opt.output) or '.', exist_ok=True)
    df.to_csv(opt.output, index=False)
    print(f"Saved {len(df)} samples × {features.shape[1]} features to {opt.output}")


if __name__ == '__main__':
    main()
