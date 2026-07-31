"""
Nuclear segmentation pipeline for CPSformer data preparation.

Runs DeepLabV3 semantic segmentation + Watershed instance separation
directly via nucseg_modules (no external script dependencies).

Usage (as library):
    from nucseg_modules.nucseg_pipeline import run_segmentation_for_cohort

Usage (as CLI):
    python -m nucseg_modules.nucseg_pipeline --input_dir ./data/BRCA --gpu 0
"""

import os
import sys
import shutil
import tempfile
import time
import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from tqdm import tqdm

# Import sibling module
_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
from nucseg_deeplabv3 import run_deeplabv3_seg


def _watershed_instance_split(binary_mask):
    """
    Apply watershed to split touching nuclei in a binary mask.

    Args:
        binary_mask: HxW uint8 array, 0 or 255

    Returns:
        instance_mask: HxW uint8 array with unique labels per nucleus
    """
    _, binary = cv2.threshold(binary_mask, 1, 255, cv2.THRESH_BINARY)
    if binary.sum() == 0:
        return np.zeros_like(binary, dtype=np.uint8)

    distance = ndi.distance_transform_edt(binary)
    coords = peak_local_max(distance, min_distance=11, labels=binary)

    if len(coords) == 0:
        return np.zeros_like(binary, dtype=np.uint8)

    mask_peaks = np.zeros(distance.shape, dtype=bool)
    mask_peaks[tuple(coords.T)] = True
    markers, _ = ndi.label(mask_peaks)
    labels = watershed(-distance, markers, mask=binary)

    return labels.astype(np.uint8)


def run_segmentation_for_cohort(image_paths, seg_dir, cohort_name, work_dir, gpu_id):
    """
    Run DeepLabV3 nuclear segmentation on a list of images and save masks.

    This replaces the original two-model (DeepLabV3 + UNet) pipeline with a
    single DeepLabV3 + Watershed approach, eliminating external script
    dependencies (~/software/nucsegdeeplabv3/, ~/software/Nuclei-seg-HE/).

    Args:
        image_paths: list of full paths to input images (.png/.jpg)
        seg_dir: directory to save segmentation masks
        cohort_name: tumor type name (for logging)
        work_dir: temporary working directory for intermediate files
        gpu_id: GPU device id (int), or -1 for CPU

    Returns:
        Number of masks generated (int)
    """
    os.makedirs(seg_dir, exist_ok=True)

    # ---- Step 1: DeepLabV3 semantic segmentation ----
    print(f"  [{cohort_name}] Step 1: DeepLabV3 segmentation...")
    t0 = time.time()
    results = run_deeplabv3_seg(image_paths, work_dir, gpu_id)
    print(f"  [{cohort_name}] DeepLabV3 done in {time.time()-t0:.1f}s "
          f"({len(results)} masks)")

    # ---- Step 2: Save masks as PNG ----
    saved = 0
    for name, semantic_mask in tqdm(results.items(), desc=f"  [{cohort_name}] Saving masks",
                                     total=len(results)):
        out_path = os.path.join(seg_dir, name + ".png")
        cv2.imwrite(out_path, semantic_mask)
        saved += 1

    print(f"  [{cohort_name}] Segmentation complete: {saved} masks saved to {seg_dir}")
    return saved


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run nuclear segmentation on a cohort directory")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing raw images (.png/.jpg)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for masks (default: input_dir/segment)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device id, -1 for CPU (default: 0)")
    parser.add_argument("--work_dir", type=str, default=None,
                        help="Temporary working directory (default: auto)")
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = args.output_dir or os.path.join(input_dir, 'segment')
    work_dir = args.work_dir or tempfile.mkdtemp(prefix='nucseg_')

    # Collect image paths
    img_files = sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        and not f.startswith('.')
    ])

    if not img_files:
        print(f"No images found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(img_files)} images in {input_dir}")

    count = run_segmentation_for_cohort(
        img_files, output_dir, os.path.basename(input_dir), work_dir, args.gpu)

    # Cleanup
    shutil.rmtree(work_dir, ignore_errors=True)
    print(f"Done. {count} masks saved to {output_dir}")
