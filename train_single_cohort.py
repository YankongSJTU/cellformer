# train_single_cohort.py - 单卡全量训练 (cohort_pkls + SupConLoss)
# 基于已验证的 alias 启动方式, 无需 DDP/NCCL
#
# 大 batch 单卡训练支持:
#   1. cell_encoder 在 torch.no_grad() 下前向 (frozen, 省激活显存, 见 models.py)
#   2. Transformer 梯度检查点 (--gradient_checkpointing, 省 O(B*N^2) 激活)
#   3. 梯度累积 (--accum_steps, 有效 batch = batch_size * accum_steps)
#   4. max_cells 可调 (--max_cells, 降 N 以换更大 B)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import pickle
import argparse
import random
from glob import glob
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, ConcatDataset
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from models import MILCellModelmerge, random_subgraph_crop
from utils.DataSets import DatasetLoaderV2
from utils.utils import NTXentLoss, SupConLoss, cal_loss5, custom_collate_fn, mask_cell_features



def parse_args():
    parser = argparse.ArgumentParser(description="CPSformer Single-GPU Training with Cohort Data")
    parser.add_argument('--pkl_dir', type=str, required=True,
                        help='Directory containing per-cohort pkl files')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints_supcon')
    parser.add_argument('--pretrained_model_path', default='./checkpoints_merged_v2/best_model.pth')
    parser.add_argument('--distilled_cell_path', default='./checkpoints_cell/model.pth')
    parser.add_argument('--epoch_count', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--featuredim', type=int, default=1024)
    parser.add_argument('--dmodel', type=int, default=256)
    parser      .add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--alpha', type=float, default=0.1, help='instance loss weight')
    parser.add_argument('--gamma', type=float, default=0.1, help='diversity loss weight')
    parser.add_argument('--delta', type=float, default=0.8, help='cls loss weight')
    parser.add_argument('--beta', type=float, default=0.1, help='discriminative (SupCon) loss weight')
    parser.add_argument('--temp', type=float, default=0.1, help='contrastive temperature')
    parser.add_argument('--gpu_id', type=int, default=0, help='CUDA device id')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--crop_min_frac', type=float, default=0.3)
    parser.add_argument('--crop_max_frac', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--accum_steps', type=int, default=1,
                        help='Gradient accumulation steps; effective batch = batch_size * accum_steps')
    parser.add_argument('--max_cells', type=int, default=2500,
                        help='Max cells per bag (cap to reduce memory for larger batch)')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='Enable gradient checkpointing on Transformer to save memory')
    parser.add_argument('--encoder_chunk_size', type=int, default=0,
                        help='Chunk size for cell_encoder forward; limits peak GPU memory')
    return parser.parse_args()


def build_dataset(opt):
    """加载所有 cohort pkl, 划分 train/val, 合并为 ConcatDataset."""
    pkl_files = sorted(glob(os.path.join(opt.pkl_dir, '*.pkl')))
    # 过滤旧格式文件
    pkl_files = [f for f in pkl_files if not os.path.basename(f).startswith('data')]
    print(f'Found {len(pkl_files)} cohort pkl files')

    all_tr_datasets, all_va_datasets = [], []
    num_classes = 24

    for pkl_path in pkl_files:
        name = os.path.basename(pkl_path)
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        n = len(data['x_imgname'])
        indices = np.arange(n)
        tr_idx, va_idx = train_test_split(
            indices, test_size=opt.val_split, random_state=opt.seed, shuffle=True)

        def subset(d, idx_list):
            return {k: [v[i] for i in idx_list] for k, v in d.items()}

        tr_data = subset(data, tr_idx)
        va_data = subset(data, va_idx)

        all_tr_datasets.append(DatasetLoaderV2(tr_data, is_train=True, max_cells=opt.max_cells))
        all_va_datasets.append(DatasetLoaderV2(va_data, is_train=False, max_cells=opt.max_cells))
        del data, tr_data, va_data
        print(f'  {name}: {n} samples ({len(tr_idx)} train / {len(va_idx)} val)')

    tr_dataset = ConcatDataset(all_tr_datasets)
    va_dataset = ConcatDataset(all_va_datasets)
    print(f'Dataset: train={len(tr_dataset)}, val={len(va_dataset)}, classes={num_classes}')
    return tr_dataset, va_dataset, num_classes


def run_epoch(loader, model, optimizer, scaler, ntxent_loss, supcon_loss, cls_criterion, opt, epoch, is_train=True, log_path=None):
    model.train(is_train)
    torch.set_grad_enabled(is_train)
    epoch_loss = 0.0
    count = 0
    desc = f'Epoch {epoch} [{"Train" if is_train else "Val"}]'
    # 梯度累积仅训练时生效; 验证不做累积
    accum_steps = opt.accum_steps if is_train else 1

    if is_train:
        optimizer.zero_grad(set_to_none=True)

    n_batches = len(loader)
    for batch_idx, (x_patches, x_masks, x_names, labels, pos) in enumerate(tqdm(loader, desc=desc)):
        if x_patches.numel() == 0 or x_masks.numel() == 0 or pos.numel() == 0:
            continue
        x_patches, x_masks, pos, labels = x_patches.cuda(), x_masks.cuda(), pos.cuda(), labels.cuda()

        # View 1: cell-dropout
        v1_p, v1_pos, v1_m = mask_cell_features(x_patches, pos, x_masks)
        # View 2: random subgraph crop
        v2_p, v2_pos, v2_m = random_subgraph_crop(
            x_patches, pos, x_masks,
            min_frac=opt.crop_min_frac,
            max_frac=opt.crop_max_frac)

        # 释放原始 batch 数据, 两个 view 已独立拷贝
        del x_patches, x_masks, pos

        if torch.isnan(v1_p).any() or torch.isnan(v2_p).any():
            continue

        with autocast():
            feat1, _, logits1 = model(v1_p, v1_pos, v1_m)
            # v1 forward 完成, 释放 v1 输入腾显存给 v2 forward
            del v1_p, v1_pos, v1_m
            feat2, _, _ = model(v2_p, v2_pos, v2_m)
            del v2_p, v2_pos, v2_m
            if torch.isnan(feat1).any() or torch.isnan(feat2).any():
                continue

            l_con, _, _, _, l_div, l_ins = cal_loss5(feat1, feat2, ntxent_loss)
            l_cls = cls_criterion(logits1, labels)
            # 判别损失
            all_feat = torch.cat([feat1, feat2], dim=0)
            all_lab = torch.cat([labels, labels], dim=0)
            l_dis = supcon_loss(all_feat, all_lab)

            loss = ((1 - opt.alpha - opt.gamma - opt.beta) * l_con
                    + opt.gamma * l_div
                    + opt.alpha * l_ins
                    + opt.delta * l_cls
                    + opt.beta * l_dis)

            if torch.isnan(loss):
                continue

            # 梯度累积: loss 除以累积步数, 使梯度等于各 micro-batch 梯度的平均
            loss_scaled = loss / accum_steps

        if is_train:
            scaler.scale(loss_scaled).backward()

            # 每 accum_steps 个 micro-batch (或 epoch 最后一个不完整组) 才更新一次
            should_step = ((batch_idx + 1) % accum_steps == 0) or ((batch_idx + 1) == n_batches)
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if count % 50 == 0 and log_path:
                with open(log_path, 'a') as f:
                    f.write(f'{epoch},{batch_idx},{loss.item():.4f},{l_con.item():.4f},{l_cls.item():.4f},{l_dis.item():.4f}\n')
        else:
            # 验证: 不反传, 直接累加原始 loss
            pass

        epoch_loss += loss.item()
        count += 1
    return epoch_loss / count if count > 0 else 0.0


def main():
    opt = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(opt.gpu_id)
    os.makedirs(opt.checkpoints_dir, exist_ok=True)
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    eff_batch = opt.batch_size * opt.accum_steps
    print('=' * 60)
    print(f'CPSformer Single-GPU Large-Batch Training')
    print(f'  GPU: cuda:{opt.gpu_id}, batch_size={opt.batch_size}, accum_steps={opt.accum_steps}')
    print(f'  effective_batch={eff_batch}, max_cells={opt.max_cells}')
    print(f'  grad_checkpoint={opt.gradient_checkpointing}, epochs={opt.epoch_count}, lr={opt.lr}')
    print('=' * 60)

    # 1. 加载数据
    tr_dataset, va_dataset, num_classes = build_dataset(opt)

    tr_loader = DataLoader(tr_dataset, batch_size=opt.batch_size, shuffle=True,
                            num_workers=opt.num_workers, collate_fn=custom_collate_fn, drop_last=True)
    va_loader = DataLoader(va_dataset, batch_size=opt.batch_size, shuffle=False,
                            num_workers=opt.num_workers, collate_fn=custom_collate_fn)

    # 2. 模型
    model = MILCellModelmerge(
        num_classes=num_classes, d_model=opt.dmodel,
        output_dim=opt.featuredim,
        distilled_path=opt.distilled_cell_path,
        use_gradient_checkpointing=opt.gradient_checkpointing,
        encoder_chunk_size=opt.encoder_chunk_size,
    ).cuda()

    if os.path.exists(opt.pretrained_model_path):
        print(f'Loading pretrained: {opt.pretrained_model_path}')
        ckpt = torch.load(opt.pretrained_model_path, map_location='cuda')
        sd = ckpt.get('model_state_dict', ckpt)
        new_sd = {k.replace('module.', ''): v for k, v in sd.items()
                  if not k.replace('module.', '').startswith('cell_encoder.')}
        model.load_state_dict(new_sd, strict=False)
        print('Pretrained structural weights loaded.')

    total_p = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total params: {total_p:,}, Trainable: {trainable_p:,}')

    # 3. 优化器
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=opt.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epoch_count)
    scaler = GradScaler()
    # NTXentLoss forward 内部按实际 batch 动态构造掩码, 此处 batch_size 仅占位
    ntxent = NTXentLoss(batch_size=opt.batch_size, temperature=opt.temp, device='cuda')
    supcon = SupConLoss(temperature=opt.temp).cuda()
    cls_crit = nn.CrossEntropyLoss()

    # 4. 训练
    log_path = os.path.join(opt.checkpoints_dir, 'train_log.csv')
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write('epoch,train_loss,l_con,l_cls,l_dis,val_loss,v_con,v_cls,v_dis,lr\n')

    best_val = float('inf')
    for epoch in range(opt.epoch_count):
        tr_loss = run_epoch(tr_loader, model, optimizer, scaler, ntxent, supcon, cls_crit, opt, epoch, True, log_path)
        val_loss = run_epoch(va_loader, model, optimizer, scaler, ntxent, supcon, cls_crit, opt, epoch, False)
        scheduler.step()
        cur_lr = optimizer.param_groups[0]['lr']

        print(f'Epoch {epoch:3d}/{opt.epoch_count}  lr={cur_lr:.2e}  '
              f'train={tr_loss:.4f}  val={val_loss:.4f}')

        with open(log_path, 'a') as f:
            f.write(f'{epoch},{tr_loss:.4f},0,0,0,{val_loss:.4f},0,0,0,{cur_lr:.2e}\n')

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'best_val': best_val,
            }, os.path.join(opt.checkpoints_dir, 'best_model.pth'))
            print(f'  -> Saved best model (val_loss={val_loss:.4f})')

        if (epoch + 1) % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            }, os.path.join(opt.checkpoints_dir, f'checkpoint_epoch{epoch}.pth'))

    print(f'\nTraining complete. Best val_loss={best_val:.4f}')


if __name__ == '__main__':
    main()
