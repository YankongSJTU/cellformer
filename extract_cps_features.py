#!/usr/bin/env python3
"""
Extract CPS features using trained CPSformer model for downstream tasks.
Usage: python extract_cps_features.py --cohort BRCA --gpu 0 --samples_per_patient 50
"""

import os
import sys
import cv2
import numpy as np
import torch
import pickle
import argparse
import random
import tempfile
import shutil
from tqdm import tqdm
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import pandas as pd

# Add project path
sys.path.insert(0, '/export/home/kongyan/project/newcellformer')
from models import MILCellModelmerge


def _ensure_segmentation(img_files, seg_dir, cohort_name, gpu_id='0'):
    """
    Run DeepLabV3 segmentation for images that have no corresponding mask.

    Args:
        img_files: list of image filenames in img_dir
        seg_dir: directory where masks should be
        cohort_name: for logging
        gpu_id: GPU device id string
    """
    missing = []
    for f in img_files:
        base = os.path.splitext(f)[0]
        mask_png = os.path.join(seg_dir, base + '.png')
        mask_jpg = os.path.join(seg_dir, base + '.jpg')
        if not os.path.exists(mask_png) and not os.path.exists(mask_jpg):
            missing.append(f)

    if not missing:
        print(f"  [{cohort_name}] All masks present, skipping segmentation.")
        return

    print(f"  [{cohort_name}] {len(missing)} images missing masks, running segmentation...")

    # Build full paths for missing images
    img_dir = os.path.dirname(os.path.join(seg_dir, '..', 'image'))
    missing_paths = [os.path.join(img_dir, f) for f in missing]

    import nucseg_modules
    nucseg_root = os.path.dirname(nucseg_modules.__file__)

    # Import and run segmentation
    sys.path.insert(0, nucseg_root)
    from nucseg_deeplabv3 import run_deeplabv3_seg

    work_dir = tempfile.mkdtemp(prefix='cps_seg_')
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        results = run_deeplabv3_seg(missing_paths, work_dir, int(gpu_id))

        saved = 0
        for name, mask in results.items():
            out_path = os.path.join(seg_dir, name + '.png')
            cv2.imwrite(out_path, mask)
            saved += 1
        print(f"  [{cohort_name}] Generated {saved}/{len(missing)} masks.")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract CPS features for downstream tasks")
    parser.add_argument('--root_dir', type=str,
                        default='/export/home/kongyan/project/cellformer/data',
                        help="Root directory containing tumor type folders")
    parser.add_argument('--cohort', type=str, default='all',
                        help="Specific cohort to process (e.g., 'BRCA', 'LUAD'), or 'all'")
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    parser.add_argument('--model_path', type=str,
                        default='/export/home/kongyan/project/newcellformer/checkpoints_zero/best_distilled_model.pth',
                        help="Path to trained CPSformer model")
    parser.add_argument('--distilled_cell_path', type=str,
                        default='/export/home/kongyan/project/newcellformer/checkpoints_cell/model.pth',
                        help="Path to distilled cell encoder")
    parser.add_argument('--output_dir', type=str,
                        default='/export/home/kongyan/project/newcellformer/features_cpsformer',
                        help="Output directory for feature files")
    parser.add_argument('--samples_per_patient', type=int, default=50,
                        help="Max number of patches to sample per patient (for efficiency)")
    parser.add_argument('--max_cells', type=int, default=500,
                        help="Max cells to process per image (for memory)")
    parser.add_argument('--patch_size', type=int, default=56,
                        help="Cell patch size (56x56)")
    parser.add_argument('--auto_segment', action='store_true',
                        help='Auto-run DeepLabV3 segmentation for images missing masks')
    parser.add_argument('--batch_size', type=int, default=1,
                        help="Batch size for inference")
    return parser.parse_args()


def load_model(opt):
    """Load trained CPSformer model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MILCellModelmerge(
        num_classes=24,
        d_model=256,
        output_dim=1024,
        distilled_path=opt.distilled_cell_path
    ).to(device)

    if os.path.exists(opt.model_path):
        checkpoint = torch.load(opt.model_path, map_location=device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

        # Filter out cell_encoder weights (already loaded via distilled_path)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            if name.startswith('cell_encoder.'):
                continue
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded model from {opt.model_path}")

    model.eval()
    return model, device


def extract_cells_from_segment(img_rgb, seg_mask, patch_size=56, max_cells=500):
    """
    Extract cell patches and positions from image using watershed segmentation.

    Args:
        img_rgb: RGB image array (H, W, 3)
        seg_mask: Binary segmentation mask (H, W)
        patch_size: Size of each cell patch
        max_cells: Maximum cells to extract

    Returns:
        patches: array of cell patches (N, 3, patch_size, patch_size)
        positions: array of cell centroids (N, 2) in (x, y) format
        n_cells: number of cells extracted
    """
    # Threshold
    _, binary_mask = cv2.threshold(seg_mask, 1, 255, cv2.THRESH_BINARY)

    # Distance transform
    distance = ndi.distance_transform_edt(binary_mask)

    # Find local maxima (nuclei centers)
    coords = peak_local_max(distance, min_distance=11, labels=binary_mask)

    if len(coords) == 0:
        return None, None, 0

    # Limit cells
    if len(coords) > max_cells:
        idx = np.random.choice(len(coords), max_cells, replace=False)
        coords = coords[idx]

    n_cells = len(coords)
    patches = []
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

        patches.append(crop)
        positions.append([x, y])  # Store as (x, y)

    patches = np.array(patches, dtype=np.float32)
    # Normalize and convert to tensor format (N, 3, H, W)
    patches = patches.transpose(0, 3, 1, 2) / 255.0
    positions = np.array(positions, dtype=np.float32)

    return patches, positions, n_cells


def extract_feature_from_image(model, device, img_path, seg_path, opt):
    """
    Extract CPS feature from a single image.

    Returns:
        feature: 1024-dim numpy array
        sample_name: patient ID (first 12 chars of filename)
        img_name: full image filename
    """
    img_bgr = cv2.imread(img_path)
    seg = cv2.imread(seg_path, 0)

    if img_bgr is None or seg is None:
        return None, None, None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Extract cells
    patches, positions, n_cells = extract_cells_from_segment(
        img_rgb, seg, opt.patch_size, opt.max_cells
    )

    if n_cells < 15:  # Minimum cells threshold
        return None, None, None

    # Prepare tensors
    patches_tensor = torch.from_numpy(patches).unsqueeze(0).to(device)  # (1, N, 3, H, W)
    pos_tensor = torch.from_numpy(positions).unsqueeze(0).to(device)    # (1, N, 2)
    mask_tensor = torch.ones(1, n_cells).to(device)                      # (1, N)

    # Inference
    with torch.no_grad():
        feature, _, _ = model(patches_tensor, pos_tensor, mask_tensor)

    feature = feature.cpu().numpy().flatten()

    # Extract sample name (patient ID)
    img_name = os.path.basename(img_path)
    sample_name = img_name[:12]  # TCGA patient ID format

    return feature, sample_name, img_name


def process_cohort(cohort_name, model, device, opt):
    """
    Process a single tumor cohort and save features.
    """
    cohort_dir = os.path.join(opt.root_dir, f'data{cohort_name}')
    img_dir = os.path.join(cohort_dir, 'image')
    seg_dir = os.path.join(cohort_dir, 'segment')

    if not (os.path.exists(img_dir) and os.path.exists(seg_dir)):
        print(f"Skipping {cohort_name}: directories not found")
        return

    # Get all image files
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]

    if len(img_files) == 0:
        print(f"Skipping {cohort_name}: no images found")
        return

    print(f"Processing {cohort_name}: {len(img_files)} images")

    # Auto-segment if enabled and masks are missing
    if opt.auto_segment:
        if not os.path.exists(seg_dir):
            os.makedirs(seg_dir, exist_ok=True)
        _ensure_segmentation(img_files, seg_dir, cohort_name, opt.gpu)

    # Group images by patient for sampling
    patient_images = {}
    for f in img_files:
        patient_id = f[:12]
        if patient_id not in patient_images:
            patient_images[patient_id] = []
        patient_images[patient_id].append(f)

    # Sample images per patient
    sampled_images = []
    for patient_id, files in patient_images.items():
        if len(files) > opt.samples_per_patient:
            sampled = random.sample(files, opt.samples_per_patient)
        else:
            sampled = files
        sampled_images.extend(sampled)

    print(f"  Sampled {len(sampled_images)} images from {len(patient_images)} patients")

    # Extract features
    features_list = []
    sample_names_list = []
    img_names_list = []

    for img_fname in tqdm(sampled_images, desc=cohort_name):
        img_path = os.path.join(img_dir, img_fname)

        # Find corresponding segment
        base_name = os.path.splitext(img_fname)[0]
        seg_path = os.path.join(seg_dir, base_name + '.png')
        if not os.path.exists(seg_path):
            seg_path = os.path.join(seg_dir, base_name + '.jpg')

        if not os.path.exists(seg_path):
            continue

        feature, sample_name, img_name = extract_feature_from_image(
            model, device, img_path, seg_path, opt
        )

        if feature is not None:
            features_list.append(feature)
            sample_names_list.append(sample_name)
            img_names_list.append(img_name)

    if len(features_list) == 0:
        print(f"  No valid features extracted for {cohort_name}")
        return

    # Save features
    features_array = np.array(features_list)

    # Create DataFrame
    feature_cols = [str(i) for i in range(features_array.shape[1])]
    df = pd.DataFrame(features_array, columns=feature_cols)
    df['samplename'] = sample_names_list
    df['imgname'] = img_names_list

    output_path = os.path.join(opt.output_dir, f'{cohort_name}.cps_feature.csv')
    df.to_csv(output_path, index=False)

    print(f"  Saved {len(features_list)} features to {output_path}")
    print(f"  Feature shape: {features_array.shape}")

    return df


def main():
    opt = parse_args()
    random.seed(42)

    os.makedirs(opt.output_dir, exist_ok=True)

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

    # Load model
    model, device = load_model(opt)

    # Define cohorts to process
    all_cohorts = [
        'BLCA', 'BRCA', 'CESC', 'COAD', 'DLBC', 'ESCA', 'GBM', 'HNSC',
        'KIRC', 'KIRP', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'OV', 'PAAD',
        'PRAD', 'READ', 'STAD', 'THCA', 'THYM', 'UCEC'
    ]

    if opt.cohort == 'all':
        cohorts = all_cohorts
    else:
        cohorts = [opt.cohort.upper()]

    print(f"Processing cohorts: {cohorts}")

    for cohort in cohorts:
        process_cohort(cohort, model, device, opt)

    print("Done!")


if __name__ == "__main__":
    main()