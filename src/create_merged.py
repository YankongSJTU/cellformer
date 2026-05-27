#!/usr/bin/env python3
"""Create merged training data from semantic segmentation masks using watershed.

Reads image/ + segment/ directories for each TCGA cohort, applies watershed
algorithm to separate touching nuclei, extracts centroids and 56x56 patches,
then samples per tumor type and saves as a single pkl file.
"""
import argparse
import os
import pickle
import random
import sys
import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from tqdm import tqdm
from multiprocessing.pool import Pool

TRAINING_COHORTS = [
    "dataBLCA", "dataBRCA", "dataCESC", "dataCOAD", "dataDLBC", "dataESCA",
    "dataGBM", "dataHNSC", "dataKIRC", "dataKIRP", "dataLGG", "dataLIHC",
    "dataLUAD", "dataLUSC", "dataOV", "dataPAAD", "dataPRAD", "dataREAD",
    "dataSTAD", "dataTHCA", "dataTHYM", "dataUCEC1",
]

PATCH_SIZE = 56
CROP_RADIUS = 40

def watershed_nuclei(binary_mask, min_distance=7):
    """Separate touching nuclei using watershed on distance transform.

    Args:
        binary_mask: uint8 array, 0=background, 255=nuclei
        min_distance: minimum distance between watershed peaks

    Returns:
        instance_labels: int array, each unique value = one nucleus instance
        centroids: (N, 2) array of (x, y) centroids
    """
    mask = (binary_mask > 0).astype(np.uint8)
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=np.int32), np.zeros((0, 2), dtype=np.float32)

    distance = ndi.distance_transform_edt(mask)
    coords = peak_local_max(distance, min_distance=min_distance, labels=mask)
    local_max_mask = np.zeros(distance.shape, dtype=bool)
    local_max_mask[tuple(coords.T)] = True
    markers, _ = ndi.label(local_max_mask)
    labels = watershed(-distance, markers, mask=mask)

    # Extract centroids from instance labels
    max_label = labels.max()
    centroids = []
    for i in range(1, max_label + 1):
        ys, xs = np.where(labels == i)
        if len(xs) == 0:
            continue
        cx, cy = np.mean(xs), np.mean(ys)
        centroids.append([cx, cy])

    if len(centroids) == 0:
        return labels, np.zeros((0, 2), dtype=np.float32)

    return labels, np.array(centroids, dtype=np.float32)


def process_single_image(args):
    """Process one image + segment pair: watershed + patch extraction."""
    img_path, seg_path, tumor_type, max_cells = args

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    seg_gray = cv2.imread(seg_path, 0)
    if seg_gray is None:
        return None
    _, binary_mask = cv2.threshold(seg_gray, 1, 255, cv2.THRESH_BINARY)

    _, centroids = watershed_nuclei(binary_mask, min_distance=7)
    if len(centroids) == 0:
        return None

    # Subsample if too many cells
    if len(centroids) > max_cells:
        idx = np.random.choice(len(centroids), max_cells, replace=False)
        centroids = centroids[idx]

    # Extract patches
    h, w = img_rgb.shape[:2]
    patches = []
    final_coords = []

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
            crop = cv2.resize(crop, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_CUBIC)

        patches.append(crop)
        final_coords.append([cx, cy])

    pat_name = os.path.basename(img_path)[:12]

    return {
        "sample_name": pat_name,
        "img_name": os.path.basename(img_path),
        "nuc_patches": np.array(patches, dtype=np.uint8),
        "nuc_pos": np.array(final_coords, dtype=np.float32),
        "nuc_count": len(final_coords),
        "tumor_type": tumor_type,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Create merged training data with watershed")
    #parser.add_argument('--root_dir', type=str , default='../data')
    parser.add_argument('--root_dir', type=str , default='../data')
    parser.add_argument('--save_path', type=str, default='../data/merged_Demo_train_data.pkl')
    parser.add_argument('--samples_per_type', type=int, default=10)
    parser.add_argument('--max_cells', type=int, default=1500)
    parser.add_argument('--max_images', type=int, default=20, help='Max images to process per cohort (speeds up creation)')
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--cohorts', type=str, nargs='*', default=None, help='Specific cohorts to process (default: all)')
    return parser.parse_args()


def main():
    opt = parse_args()
    random.seed(42)
    np.random.seed(42)

    cohorts = opt.cohorts if opt.cohorts else TRAINING_COHORTS

    print(f"Watershed-based data creation")
    print(f"Root: {opt.root_dir}")
    print(f"Cohorts: {len(cohorts)}")
    print(f"Samples per type: {opt.samples_per_type}")
    print(f"Max cells per image: {opt.max_cells}")
    print(f"Workers: {opt.num_workers}")

    all_results = {}

    for cohort in tqdm(cohorts, desc="Cohorts"):
        cohort_dir = os.path.join(opt.root_dir, cohort)
        tumor_type = cohort.replace('data', '')
        image_dir = os.path.join(cohort_dir, 'image')
        segment_dir = os.path.join(cohort_dir, 'segment')

        if not os.path.isdir(image_dir) or not os.path.isdir(segment_dir):
            print(f"  [{tumor_type}] Missing image/ or segment/, skipping.")
            continue

        # Match images that have corresponding segment files
        img_files = sorted([f for f in os.listdir(image_dir)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        tasks = []
        for fname in img_files:
            if len(tasks) >= opt.max_images:
                break
            base = os.path.splitext(fname)[0]
            seg_path = os.path.join(segment_dir, base + '.png')
            if not os.path.exists(seg_path):
                seg_path = os.path.join(segment_dir, base + '.jpg')
            if not os.path.exists(seg_path):
                continue
            img_path = os.path.join(image_dir, fname)
            tasks.append((img_path, seg_path, tumor_type, opt.max_cells))

        if not tasks:
            print(f"  [{tumor_type}] No matched image-segment pairs.")
            continue

        print(f"  [{tumor_type}] Processing {len(tasks)} images...")

        pool = Pool(processes=opt.num_workers)
        results = list(tqdm(pool.imap(process_single_image, tasks),
                           total=len(tasks), desc=f"  {tumor_type}"))
        pool.close()
        pool.join()

        valid = [r for r in results if r is not None]
        print(f"  [{tumor_type}] {len(valid)}/{len(tasks)} images with cells")

        # Sample per tumor type
        if len(valid) > opt.samples_per_type:
            valid = random.sample(valid, opt.samples_per_type)

        all_results[tumor_type] = valid

    # Merge all
    merged = {
        'x_samplename': [],
        'x_imgname': [],
        'x_nucpatch': [],
        'x_nucpatch_pos': [],
        'x_nucpatch_no': [],
        'x_tumor': [],
    }

    total = 0
    for tumor_type, results in all_results.items():
        for r in results:
            merged['x_samplename'].append(r['sample_name'])
            merged['x_imgname'].append(r['img_name'])
            merged['x_nucpatch'].append(r['nuc_patches'])
            merged['x_nucpatch_pos'].append(r['nuc_pos'])
            merged['x_nucpatch_no'].append(r['nuc_count'])
            merged['x_tumor'].append(r['tumor_type'])
        total += len(results)
        print(f"  {tumor_type}: {len(results)} images")

    print(f"\nTotal: {total} images")

    os.makedirs(os.path.dirname(opt.save_path), exist_ok=True)
    print(f"Saving to {opt.save_path}...")
    with open(opt.save_path, 'wb') as f:
        pickle.dump(merged, f, protocol=4)

    fsize = os.path.getsize(opt.save_path) / (1024**3)
    print(f"Done! File size: {fsize:.1f} GB")


if __name__ == '__main__':
    main()
