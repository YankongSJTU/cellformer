"""
DeepLabV3-based nuclear semantic segmentation module.

Wraps the nucseg1000 pipeline into a callable function:
  split_image_into_patches -> model_inference -> merge_patches -> output mask

Dependencies: PyTorch, torchvision, albumentations, segmentation_models_pytorch
"""

import cv2
import os
import sys
import shutil
import numpy as np
from glob import glob
from PIL import Image

import torch
import torch.nn as nn
import yaml
import albumentations
from albumentations.augmentations import transforms as A_transforms
from albumentations.core.composition import Compose
from tqdm import tqdm

# Import sibling modules
_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
from deeplabv3_dataset import TestDataset
from deeplabv3_utils import conform, fillimage, cropimg


def run_deeplabv3_seg(image_paths, work_dir, gpu_id=0):
    """
    Run DeepLabV3 nuclear segmentation on a list of images.

    Args:
        image_paths: list of full paths to input images (.png/.jpg)
        work_dir: temporary working directory for this step
        gpu_id: GPU device id (int), or -1 for CPU

    Returns:
        dict mapping basename -> anno mask (numpy HxW, uint8, 0 or 255)
    """
    # CUDA_VISIBLE_DEVICES is set by the caller (prepare_data.py)

    # Locate model config and weights relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(project_root, 'checkpoints', 'nucseg_deeplabv3', 'models')

    with open(os.path.join(model_dir, 'config.yml'), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Setup temporary directories
    tmp_input_dir = os.path.join(work_dir, 'deeplabv3_tmpinputs')
    tmp_result_dir = os.path.join(work_dir, 'deeplabv3_result')
    piece_dir = os.path.join(tmp_result_dir, 'segmentpiece')
    for d in [tmp_input_dir, tmp_input_dir + '/images', tmp_result_dir, piece_dir]:
        os.makedirs(d, exist_ok=True)

    # ---- Step 1: Split images into patches ----
    img_ids = []
    for line in image_paths:
        pil_img = Image.open(line)
        img = np.array(pil_img)
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        filename = os.path.join(tmp_input_dir, 'images',
                                os.path.basename(os.path.splitext(line)[0]))
        (h, w, c) = img.shape
        k = 0
        if h > 1000:
            for i in range(int((h - 1000) / 990) + 1):
                for j in range(int((w - 1000) / 990) + 1):
                    tmpimg = img[i * 990:(i * 990 + 1000), j * 990:(j * 990 + 1000)]
                    cv2.imwrite(filename + "_" + str(k) + ".png", tmpimg)
                    k += 1
                tmpimg = img[i * 990:(i * 990 + 1000), (w - 1000):w]
                cv2.imwrite(filename + "_" + str(k) + ".png", tmpimg)
                k += 1
            for j in range(int((w - 1000) / 990) + 1):
                tmpimg = img[(h - 1000):h, j * 990:(j * 990 + 1000)]
                cv2.imwrite(filename + "_" + str(k) + ".png", tmpimg)
                k += 1
            tmpimg = img[(h - 1000):h, (w - 1000):w]
            cv2.imwrite(filename + "_" + str(k) + ".png", tmpimg)
        else:
            shutil.copy2(line, filename + "_reseize.png")

    # ---- Step 2: Build model and run inference on patches ----
    model = torch.hub.load('pytorch/vision:v0.11.2', 'deeplabv3_resnet50', pretrained=False)
    if gpu_id >= 0:
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=[0])
        model.load_state_dict(
            torch.load(os.path.join(model_dir, 'model.pth')), strict=False)
    else:
        model = nn.DataParallel(model, device_ids=[0])
        model.load_state_dict(
            torch.load(os.path.join(model_dir, 'model.pth'), map_location='cpu'), strict=True)
    model.eval()

    test_transform = Compose([
        albumentations.Resize(config['input_h'], config['input_w']),
        A_transforms.Normalize(),
    ])

    img_ids = glob(os.path.join(tmp_input_dir, 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    test_dataset = TestDataset(
        img_ids=img_ids,
        img_dir=os.path.join(tmp_input_dir, 'images'),
        img_ext=config['img_ext'],
        num_classes=config['num_classes'],
        transform=test_transform)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    with torch.no_grad():
        for input_batch, meta in tqdm(test_loader, total=len(test_loader), desc="DeepLabV3 inference"):
            if gpu_id >= 0:
                input_batch = input_batch.cuda()
            output = model(input_batch)
            output = torch.sigmoid(output['out'][:, 0:1, :, :]).cpu().numpy()
            for i in range(len(output)):
                _, tmp = cv2.threshold((output[i, 0] * 255).astype('uint8'), 220, 255, cv2.THRESH_BINARY)
                cv2.imwrite(os.path.join(piece_dir, meta['img_id'][i] + '.png'), tmp)

    if gpu_id >= 0:
        torch.cuda.empty_cache()

    # ---- Step 3: Merge patches back to full-size masks ----
    results = {}
    for line2 in image_paths:
        line2 = line2.rstrip()
        img = cv2.imread(line2)
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        (h, w, _) = img.shape

        filename_piece = os.path.join(piece_dir, os.path.basename(os.path.splitext(line2)[0]))
        filename_raw = os.path.join(tmp_input_dir, 'images', os.path.basename(os.path.splitext(line2)[0]))

        annoimg = np.zeros([h, w])
        k = 0
        if h > 1000:
            for i in range(int((h - 1000) / 990) + 1):
                for j in range(int((w - 1000) / 990) + 1):
                    tmpimg = cv2.imread(filename_piece + "_" + str(k) + ".png", 0)
                    rawimg = cv2.imread(filename_raw + "_" + str(k) + ".png", 0)
                    if rawimg is not None and np.mean(rawimg) > 200:
                        h1, w1 = rawimg.shape
                        tmpimg = np.zeros([h1, w1])
                    if tmpimg is not None:
                        annoimg[i * 990:(i * 990 + 1000), j * 990:(j * 990 + 1000)] = conform(
                            annoimg[i * 990:(i * 990 + 1000), j * 990:(j * 990 + 1000)], tmpimg)
                    k += 1
                tmpimg = cv2.imread(filename_piece + "_" + str(k) + ".png", 0)
                rawimg = cv2.imread(filename_raw + "_" + str(k) + ".png", 0)
                if rawimg is not None and np.mean(rawimg) > 200:
                    h1, w1 = rawimg.shape
                    tmpimg = np.zeros([h1, w1])
                if tmpimg is not None:
                    annoimg[i * 990:(i * 990 + 1000), (w - 1000):w] = conform(
                        annoimg[i * 990:(i * 990 + 1000), (w - 1000):w], tmpimg)
                k += 1
            for j in range(int((w - 1000) / 990) + 1):
                tmpimg = cv2.imread(filename_piece + "_" + str(k) + ".png", 0)
                rawimg = cv2.imread(filename_raw + "_" + str(k) + ".png", 0)
                if rawimg is not None and np.mean(rawimg) > 200:
                    h1, w1 = rawimg.shape
                    tmpimg = np.zeros([h1, w1])
                if tmpimg is not None:
                    annoimg[(h - 1000):h, j * 990:(j * 990 + 1000)] = conform(
                        annoimg[(h - 1000):h, j * 990:(j * 990 + 1000)], tmpimg)
                k += 1
            tmpimg = cv2.imread(filename_piece + "_" + str(k) + ".png", 0)
            rawimg = cv2.imread(filename_raw + "_" + str(k) + ".png", 0)
            if rawimg is not None and np.mean(rawimg) > 200:
                h1, w1 = rawimg.shape
                tmpimg = np.zeros([h1, w1])
            if tmpimg is not None:
                annoimg[(h - 1000):h, (w - 1000):w] = conform(
                    annoimg[(h - 1000):h, (w - 1000):w], tmpimg)
        else:
            tmpimg = cv2.imread(filename_piece + "_reseize.png", -1)
            if tmpimg is not None:
                annoimg = cropimg(tmpimg, h, w)

        basename = os.path.splitext(os.path.basename(line2))[0]
        results[basename] = annoimg.astype(np.uint8)

    # Cleanup temp dirs
    shutil.rmtree(tmp_input_dir, ignore_errors=True)
    shutil.rmtree(piece_dir, ignore_errors=True)

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="DeepLabV3 nuclear segmentation — produce binary masks from raw images")
    parser.add_argument("image_paths", nargs="+",
                        help="One or more image file paths (.png/.jpg)")
    parser.add_argument("--work_dir", type=str, default="./nucseg_work",
                        help="Temporary working directory (default: ./nucseg_work)")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU device id, -1 for CPU (default: 0)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="If set, save masks as PNG files to this directory")
    args = parser.parse_args()

    results = run_deeplabv3_seg(args.image_paths, args.work_dir, args.gpu_id)

    print(f"Segmented {len(results)} images.")

    # Optionally write masks to disk
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        for name, mask in results.items():
            out_path = os.path.join(args.output_dir, name + ".png")
            cv2.imwrite(out_path, mask)
            print(f"  wrote {out_path}")
    else:
        # Print summary
        for name, mask in results.items():
            print(f"  {name}: shape={mask.shape}, dtype={mask.dtype}, "
                  f"unique={np.unique(mask).tolist()}")
