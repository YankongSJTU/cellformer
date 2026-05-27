#!/usr/bin/env python3
"""CPSformer v2 Training Script with DataParallel + Cross-scale augmentation.

Based on newtrain.py from the original codebase, with:
- nn.DataParallel for multi-GPU (no NCCL timeout issues)
- Cross-scale subgraph crop augmentation (random_subgraph_crop)
- AdamW optimizer
- Watershed-based nuclei segmentation data
"""
import os
import sys
import argparse
import pickle
import random
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.model_selection import train_test_split

#sys.path.insert(0, '../utils/')
#sys.path.insert(0, '../')
#sys.path.insert(0, './')

from utils.models import MILCellModelmerge, random_subgraph_crop
from utils.DataSets import DatasetLoaderV2
from utils.utils import custom_collate_fn, NTXentLoss, cal_loss5, mask_cell_features


def parse_args():
    parser = argparse.ArgumentParser(description="CPSformer")
    parser.add_argument('--merged_pkl', type=str, default='../data/merged_train_watershed.pkl')
    parser.add_argument('--checkpoints_dir', type=str, default='../checkpoints/')
    parser.add_argument('--pretrained_model_path', type=str, default=None)
    parser.add_argument('--distilled_cell_path', type=str, default='../checkpoints/cell_distill/model.pth')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--featuredim', type=int, default=1024)
    parser.add_argument('--dmodel', type=int, default=256)
    parser.add_argument('--num_classes', type=int, default=24)
    parser.add_argument('--max_cells', type=int, default=2500)
    # Loss weights
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--delta', type=float, default=0.8)
    parser.add_argument('--temp', type=float, default=0.1)
    # Cross-scale crop params
    parser.add_argument('--crop_min_frac', type=float, default=0.3)
    parser.add_argument('--crop_max_frac', type=float, default=0.9)
    return parser.parse_args()


def run_epoch(loader, model, optimizer, scaler, ntxent_loss, cls_criterion,
              opt, epoch, is_train=True):
    model.train(is_train)
    torch.set_grad_enabled(is_train)
    epoch_loss = 0.0
    count = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [{'Train' if is_train else 'Val'}]")

    for batch_idx, (x_patches, x_masks, x_names, labels, pos) in enumerate(pbar):
        if x_patches.numel() == 0 or x_masks.numel() == 0 or pos.numel() == 0:
            continue

        x_patches = x_patches.cuda()
        x_masks = x_masks.cuda()
        pos = pos.cuda()
        labels = labels.cuda()

        # View 1: cell-dropout augmentation
        v1_p, v1_pos, v1_m = mask_cell_features(x_patches, pos, x_masks)
        # View 2: random spatial sub-region crop (cross-scale invariance)
        v2_p, v2_pos, v2_m = random_subgraph_crop(
            x_patches, pos, x_masks,
            min_frac=opt.crop_min_frac, max_frac=opt.crop_max_frac)

        if (torch.isnan(v1_p).any() or torch.isnan(v2_p).any() or
                torch.isnan(v1_pos).any() or torch.isnan(v2_pos).any()):
            continue

        with autocast():
            feat1, _, logits1 = model(v1_p, v1_pos, v1_m)
            feat2, _, _ = model(v2_p, v2_pos, v2_m)

            if torch.isnan(feat1).any() or torch.isnan(feat2).any():
                continue

            l_con, _, _, _, l_div, l_ins = cal_loss5(feat1, feat2, ntxent_loss)
            l_cls = cls_criterion(logits1, labels)
            total_loss = ((1 - opt.alpha - opt.gamma) * l_con
                          + opt.gamma * l_div
                          + opt.alpha * l_ins
                          + opt.delta * l_cls)

            if torch.isnan(total_loss):
                continue

        if is_train:
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

        epoch_loss += total_loss.item()
        count += 1

        with torch.no_grad():
            preds = logits1.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        if count % 10 == 0 or count == 1:
            acc = correct / total if total > 0 else 0
            tag = 'train' if is_train else 'val'
            line = (f"[E{epoch} B{count}/{len(loader)}] "
                    f"loss={total_loss.item():.4f} con={l_con.item():.4f} "
                    f"cls={l_cls.item():.4f} acc={acc:.4f} [{tag}]\n")
            batch_log = os.path.join(opt.checkpoints_dir, "batch_log.txt")
            with open(batch_log, 'a') as f:
                f.write(line)
                f.flush()

        pbar.set_postfix({"L": f"{total_loss.item():.3f}",
                          "Cls": f"{l_cls.item():.3f}",
                          "Acc": f"{correct/total:.3f}" if total > 0 else ""})

    avg_loss = epoch_loss / count if count > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def main():
    opt = parse_args()

    visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    num_gpus = len(visible_devices.split(',')) if visible_devices else 1
    use_dp = num_gpus > 1

    torch.cuda.set_device(0)
    os.makedirs(opt.checkpoints_dir, exist_ok=True)

    print(f"CPSformer v2 Training (based on newtrain.py)")
    print(f"GPUs: {num_gpus}, Batch: {opt.batch_size * num_gpus}, DataParallel: {use_dp}")
    print(f"LR: {opt.lr}, Epochs: {opt.epochs}, Optimizer: AdamW")
    print(f"Augmentation: cell_dropout + random_subgraph_crop({opt.crop_min_frac}-{opt.crop_max_frac})")
    print(f"Loss weights: alpha={opt.alpha} gamma={opt.gamma} delta={opt.delta} temp={opt.temp}")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    print(f"Loading data from {opt.merged_pkl}...")
    with open(opt.merged_pkl, 'rb') as f:
        full_data = pickle.load(f)
    print(f"Loaded {len(full_data['x_imgname'])} images")

    indices = np.arange(len(full_data['x_imgname']))
    train_idx, val_idx = train_test_split(indices, test_size=0.1,
                                          stratify=full_data['x_tumor'], random_state=42)

    def slice_dict(d, idxs):
        return {k: [v[i] for i in idxs] for k, v in d.items()}

    train_data = slice_dict(full_data, train_idx)
    val_data = slice_dict(full_data, val_idx)
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

    train_ds = DatasetLoaderV2(train_data, is_train=True, max_cells=opt.max_cells)
    val_ds = DatasetLoaderV2(val_data, is_train=False, max_cells=opt.max_cells)
    val_ds.label_map = train_ds.label_map

    train_loader = DataLoader(train_ds, batch_size=opt.batch_size * num_gpus,
                              shuffle=True, num_workers=4,
                              collate_fn=custom_collate_fn, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=opt.batch_size * num_gpus,
                            shuffle=False, num_workers=4,
                            collate_fn=custom_collate_fn, pin_memory=True)

    model = MILCellModelmerge(
        num_classes=opt.num_classes, d_model=opt.dmodel,
        output_dim=opt.featuredim, distilled_path=opt.distilled_cell_path
    ).cuda()

    if opt.pretrained_model_path and os.path.exists(opt.pretrained_model_path):
        print(f"Loading pretrained: {opt.pretrained_model_path}")
        ckpt = torch.load(opt.pretrained_model_path, map_location='cuda', weights_only=False)
        sd = ckpt.get('model_state_dict', ckpt)
        new_sd = {k.replace('module.', ''): v for k, v in sd.items()
                  if not k.replace('module.', '').startswith('cell_encoder.')}
        model.load_state_dict(new_sd, strict=False)
        print("Loaded GAT/Transformer weights; cell encoder kept as distilled.")

    if use_dp:
        model = nn.DataParallel(model, device_ids=list(range(num_gpus)))
        print(f"DataParallel on {num_gpus} GPUs")

    optimizer = optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs)
    scaler = GradScaler()

    effective_batch = opt.batch_size * num_gpus
    ntxent_loss = NTXentLoss(batch_size=effective_batch, temperature=opt.temp, device='cuda')
    cls_criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    log_path = os.path.join(opt.checkpoints_dir, "training_log.csv")
    with open(log_path, 'w') as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")

    for epoch in range(opt.epochs):
        train_loss, train_acc = run_epoch(
            train_loader, model, optimizer, scaler, ntxent_loss,
            cls_criterion, opt, epoch, is_train=True)
        val_loss, val_acc = run_epoch(
            val_loader, model, optimizer, scaler, ntxent_loss,
            cls_criterion, opt, epoch, is_train=False)

        scheduler.step()

        with open(log_path, 'a') as f:
            f.write(f"{epoch},{train_loss:.4f},{train_acc:.4f},{val_loss:.4f},{val_acc:.4f}\n")

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            raw_model = model.module if hasattr(model, 'module') else model
            torch.save({
                'epoch': epoch,
                'model_state_dict': raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc
            }, os.path.join(opt.checkpoints_dir, "best_model.pth"))
            print(f"  Saved best model (val_loss={val_loss:.4f})")

    print("Training complete.")


if __name__ == '__main__':
    main()
