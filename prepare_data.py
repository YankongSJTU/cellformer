#!/usr/bin/env python
"""
prepare_data.py — End-to-end data preparation pipeline for CPSformer training

Workflow:
    Raw images  →  Nuclear Segmentation (DeepLabV3)  →  Watershed  →  PKL

Usage:
    python prepare_data.py --input_dir ./data --save_path ./data/merged_train.pkl --gpu_id 0

The input directory should contain subdirectories named by tumor type, e.g.:
    data/
    ├── BRCA/
    │   ├── TCGA-xx-xxxx-01Z-00-DX1_1.png
    │   └── ...
    └── LUAD/
        ├── TCGA-xx-xxxx-01Z-00-DX1_1.png
        └── ...

Each subdirectory is treated as one tumor type (cohort).
"""

import argparse
import os
import pickle
import random
import shutil
import tempfile
import time
from multiprocessing.pool import Pool

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="CPSformer End-to-End Data Preparation: "
                    "Raw Images → Nuclear Segmentation → Watershed → PKL")
    parser.add_argument('--input_dir', type=str, required=True,
                        help="Root directory containing tumor-type subdirectories with images. "
                             "E.g., ./data  (must contain ./data/BRCA/*.png, ./data/LUAD/*.png, etc.)")
    parser.add_argument('--save_path', type=str, default='./data/merged_train.pkl',
                        help="Where to save the merged pkl file")
    parser.add_argument('--target_cohorts', type=str, nargs='+', default=None,
                        help="Specific cohort names to process (e.g., BRCA LUAD). "
                             "If not set, auto-detect all subdirectories in input_dir.")
    parser.add_argument('--samples_per_type', type=int, default=1000,
                        help="Number of images to sample per tumor type")
    parser.add_argument('--max_cells', type=int, default=2000,
                        help="Max cells per image (random downsample if exceeded)")
    parser.add_argument('--num_workers', type=int, default=12,
                        help="Number of parallel workers for patch extraction")
    parser.add_argument('--patch_size', type=int, default=56,
                        help="Patch size for cell crops (pixels)")
    parser.add_argument('--gpu_id', type=int, default=0,
                        help="GPU device id for segmentation models")
    parser.add_argument('--skip_segmentation', action='store_true',
                        help="Skip segmentation, assume segment/ subdirectories already exist")
    parser.add_argument('--min_cells', type=int, default=20,
                        help="Minimum number of cells to keep an image")
    return parser.parse_args()


# ===========================================================================
# Step 1: Nuclear Segmentation
# ===========================================================================

def run_segmentation_for_cohort(cohort_dir, cohort_name, work_root, gpu_id):
    """
    Run the original two-model nuclear segmentation pipeline on all images in cohort_dir.

    Runs DeepLabV3 segmentation via nucseg_modules.nucseg_pipeline
    (no external script dependencies).

    Results are saved in cohort_dir/segment/

    Args:
        cohort_dir: path to directory containing images
        cohort_name: tumor type string (e.g., "BRCA")
        work_root: temporary working directory for intermediate files
        gpu_id: GPU device id
    """
    seg_dir = os.path.join(cohort_dir, 'segment')
    seg_dir_exists = os.path.isdir(seg_dir)

    # Find all images (skip segment/ subdirectory)
    img_files = sorted([
        os.path.join(cohort_dir, f)
        for f in os.listdir(cohort_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        and not f.startswith('.')
    ])

    if not img_files:
        print(f"  [{cohort_name}] No images found in {cohort_dir}")
        return

    print(f"  [{cohort_name}] Found {len(img_files)} images, running segmentation...")

    # Create a temporary working directory for the segmentation scripts
    seg_work = os.path.join(work_root, f'seg_{cohort_name}')
    os.makedirs(seg_work, exist_ok=True)

    from nucseg_modules.nucseg_pipeline import run_segmentation_for_cohort

    t0 = time.time()
    count = run_segmentation_for_cohort(img_files, seg_dir, cohort_name, seg_work, gpu_id)
    print(f"  [{cohort_name}] Segmentation done in {time.time()-t0:.1f}s ({count} masks)")

    # Cleanup temp dir
    shutil.rmtree(seg_work, ignore_errors=True)


# ===========================================================================
# Step 2: Watershed + Patch Extraction → PKL
# ===========================================================================

def process_single_task(args):
    """
    Read original image + segmentation mask, apply watershed, extract cell patches.
    (Same logic as original CreateMergeData.py)
    """
    img_path, seg_path, tumor_type, img_fname, opt = args

    img_bgr = cv2.imread(img_path)
    mask = cv2.imread(seg_path, 0)
    if img_bgr is None or mask is None:
        return None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Watershed to split touching nuclei
    _, binary_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    distance = ndi.distance_transform_edt(binary_mask)
    coords = peak_local_max(distance, min_distance=11, labels=binary_mask)

    if len(coords) == 0:
        return None

    mask_peaks = np.zeros(distance.shape, dtype=bool)
    mask_peaks[tuple(coords.T)] = True
    markers, _ = ndi.label(mask_peaks)
    labels = watershed(-distance, markers, mask=binary_mask)

    num_cells = len(coords)
    if num_cells < opt.min_cells:
        return None

    # Random downsample if too many cells
    if num_cells > opt.max_cells:
        idx = np.random.choice(num_cells, opt.max_cells, replace=False)
        selected_centroids = coords[idx]
    else:
        selected_centroids = coords

    patches = []
    final_coords = []
    radius = opt.patch_size // 2
    h, w = img_rgb.shape[:2]

    for y, x in selected_centroids:
        y1, y2 = max(0, int(y - radius)), min(h, int(y + radius))
        x1, x2 = max(0, int(x - radius)), min(w, int(x + radius))
        crop = img_rgb[y1:y2, x1:x2]

        if crop.shape[0] < opt.patch_size or crop.shape[1] < opt.patch_size:
            pad_h = opt.patch_size - crop.shape[0]
            pad_w = opt.patch_size - crop.shape[1]
            crop = cv2.copyMakeBorder(crop, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

        if crop.shape[0] != opt.patch_size or crop.shape[1] != opt.patch_size:
            crop = cv2.resize(crop, (opt.patch_size, opt.patch_size))

        patches.append(crop)
        final_coords.append([x, y])

    return {
        "sample_name": img_fname[:12],
        "img_name": img_fname,
        "nuc_patches": np.array(patches, dtype=np.uint8),
        "nuc_pos": np.array(final_coords, dtype=np.float32),
        "nuc_count": len(final_coords),
        "tumor_type": tumor_type
    }


# ===========================================================================
# Main
# ===========================================================================

def main():
    opt = parse_args()
    random.seed(42)
    np.random.seed(42)

    input_dir = os.path.abspath(opt.input_dir)
    if not os.path.isdir(input_dir):
        print(f"Error: input directory not found: {input_dir}")
        return

    # Auto-detect cohorts if not specified
    if opt.target_cohorts is None:
        opt.target_cohorts = sorted([
            d for d in os.listdir(input_dir)
            if os.path.isdir(os.path.join(input_dir, d))
            and not d.startswith('.') and d != '__pycache__'
        ])
    print(f"Cohorts to process: {opt.target_cohorts}")
    print(f"Input directory:    {input_dir}")
    print(f"Output PKL path:    {opt.save_path}")
    print(f"GPU ID:             {opt.gpu_id}")
    print()

    # ---- Step 1: Nuclear Segmentation (per cohort) ----
    if not opt.skip_segmentation:
        work_root = tempfile.mkdtemp(prefix='cps_seg_')
        print(f"Temporary working directory: {work_root}")
        print("=" * 60)
        print("STEP 1: Nuclear Segmentation (DeepLabV3)")
        print("=" * 60)

        for cohort in opt.target_cohorts:
            cohort_dir = os.path.join(input_dir, cohort)
            run_segmentation_for_cohort(cohort_dir, cohort, work_root, opt.gpu_id)
            print()

        # Cleanup
        shutil.rmtree(work_root, ignore_errors=True)
        print("Segmentation step complete. Temporary files cleaned up.")
        print()
    else:
        print("Skipping segmentation (--skip_segmentation flag set).")
        print("Assuming segment/ subdirectories already exist.")
        print()

    # ---- Step 2: Watershed + Patch Extraction → PKL ----
    print("=" * 60)
    print("STEP 2: Watershed Cell Separation + Patch Extraction")
    print("=" * 60)

    all_tasks = []
    for cohort in opt.target_cohorts:
        cohort_dir = os.path.join(input_dir, cohort)
        if not os.path.isdir(cohort_dir):
            continue

        tumor_type = cohort
        seg_dir = os.path.join(cohort_dir, 'segment')

        if not os.path.isdir(seg_dir):
            print(f"  [{tumor_type}] No segment/ directory found, skipping.")
            continue

        img_files = sorted([
            f for f in os.listdir(cohort_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            and not f.startswith('.')
        ])

        sampled_files = random.sample(img_files, min(len(img_files), opt.samples_per_type))

        valid_count = 0
        for f in sampled_files:
            img_p = os.path.join(cohort_dir, f)
            # Look for corresponding segmentation mask
            seg_p = os.path.join(seg_dir, os.path.splitext(f)[0] + ".png")
            if not os.path.exists(seg_p):
                seg_p = os.path.join(seg_dir, os.path.splitext(f)[0] + ".jpg")
            if os.path.exists(seg_p):
                all_tasks.append((img_p, seg_p, tumor_type, f, opt))
                valid_count += 1

        print(f"  [{tumor_type}] {valid_count}/{len(sampled_files)} images with valid masks")

    print(f"\nTotal tasks for patch extraction: {len(all_tasks)}")

    if len(all_tasks) == 0:
        print("ERROR: No valid image-mask pairs found. Check that segmentation completed successfully.")
        return

    pool = Pool(processes=opt.num_workers)
    results = list(tqdm(pool.imap(process_single_task, all_tasks), total=len(all_tasks)))
    pool.close()
    pool.join()

    valid_results = [r for r in results if r is not None]
    print(f"Valid samples: {len(valid_results)} / {len(all_tasks)}")

    # Build final data bundle
    final_bundle = {
        "x_samplename": [r['sample_name'] for r in valid_results],
        "x_imgname": [r['img_name'] for r in valid_results],
        "x_nucpatch": [r['nuc_patches'] for r in valid_results],
        "x_nucpatch_pos": [r['nuc_pos'] for r in valid_results],
        "x_nucpatch_no": [r['nuc_count'] for r in valid_results],
        "x_tumor": [r['tumor_type'] for r in valid_results]
    }

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(opt.save_path)), exist_ok=True)
    print(f"\nSaving merged data to {opt.save_path}...")
    with open(opt.save_path, "wb") as f:
        pickle.dump(final_bundle, f, protocol=4)

    # Print summary
    tumor_counts = {}
    for r in valid_results:
        t = r['tumor_type']
        tumor_counts[t] = tumor_counts.get(t, 0) + 1
    print("\nData summary:")
    for t, c in sorted(tumor_counts.items()):
        print(f"  {t}: {c} samples")
    print(f"\nTotal: {len(valid_results)} samples")
    print(f"Saved to: {opt.save_path}")
    print("\nDone! You can now run training with:")
    print(f"  python newtrain.py --merged_pkl {opt.save_path}")


if __name__ == "__main__":
    main()
