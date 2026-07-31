#!/usr/bin/env python3
"""
Drug Sensitivity Prediction with ABMIL Aggregation.

Two-stage approach:
  Stage 1: Multi-task ABMIL pretraining (predict all drugs jointly)
  Stage 2: Per-drug SVR on extracted patient embeddings

Also supports mean pooling baseline for fair comparison.

Reference:
  Dawood et al. (2024) "Cancer drug sensitivity prediction from routine
  histology images" — uses ABMIL + SVR with Spearman correlation.

Usage:
  python downstream_drug_abmil.py --model_name CPSformer_v2 \
      --features_dir ./features_cpsformer_v2 --feature_dim 1024 --gpu 0

  # Mean pooling baseline
  python downstream_drug_abmil.py --model_name CPSformer_v2 \
      --features_dir ./features_cpsformer_v2 --feature_dim 1024 --aggregation mean --gpu 0
"""

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from scipy.stats import spearmanr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Dataset
# =============================================================================

class DrugABMILDataset(Dataset):
    """Variable-length bag dataset for drug sensitivity.

    Each item is a patient with:
      - patch_features: (n_patches, input_dim)
      - drug_values:   (n_drugs,) with NaN for missing drugs
      - patient_id:    str
    """

    def __init__(self, patch_dict, drug_values_df, patient_list,
                 feature_scaler=None, max_patches=100, drug_scaler=None):
        """
        Args:
            patch_dict:     {patient_id: np.array (n_patches, input_dim)}
            drug_values_df: DataFrame, index=patient_id, columns=drugs
            patient_list:   list of patient IDs for this split
            feature_scaler:  fitted StandardScaler for patch features
            max_patches:    max patches per bag
            drug_scaler:    fitted StandardScaler for drug values
        """
        self.patch_dict = patch_dict
        self.drug_values_df = drug_values_df
        self.feature_scaler = feature_scaler
        self.max_patches = max_patches
        self.drug_scaler = drug_scaler
        self.drug_names = drug_values_df.columns.tolist()

        # Filter to patients with both features and drug data
        self.patients = [p for p in patient_list
                         if p in patch_dict and p in drug_values_df.index]

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient = self.patients[idx]
        feats = self.patch_dict[patient].astype(np.float32)  # (n_patches, dim)

        # Normalize features
        if self.feature_scaler is not None:
            feats = self.feature_scaler.transform(feats)

        # Subsample if too many patches
        n_patches = feats.shape[0]
        if n_patches > self.max_patches:
            sel = np.random.choice(n_patches, self.max_patches, replace=False)
            feats = feats[sel]
            n_patches = self.max_patches

        # Drug values (NaN for missing)
        drug_vals = self.drug_values_df.loc[patient].values.astype(np.float32)
        if self.drug_scaler is not None:
            drug_vals = self.drug_scaler.transform(drug_vals.reshape(1, -1)).flatten()

        return feats, drug_vals, patient, n_patches


def drug_collate_fn(batch):
    """Collate variable-length bags with padding."""
    feats_list, drug_list, pids, npatches = zip(*batch)

    max_len = max(f.shape[0] for f in feats_list)
    batch_size = len(feats_list)
    feat_dim = feats_list[0].shape[1]

    padded = torch.zeros(batch_size, max_len, feat_dim)
    mask = torch.zeros(batch_size, max_len)

    for i, f in enumerate(feats_list):
        n = f.shape[0]
        padded[i, :n] = torch.from_numpy(f)
        mask[i, :n] = 1.0

    n_drugs = drug_list[0].shape[0]
    drug_tensor = torch.zeros(batch_size, n_drugs)
    drug_mask = torch.zeros(batch_size, n_drugs)  # 1 = valid
    for i, dv in enumerate(drug_list):
        drug_tensor[i] = torch.from_numpy(dv)
        drug_mask[i] = torch.from_numpy(~np.isnan(dv) &
                                          np.isfinite(dv)).float()

    return padded, mask, drug_tensor, drug_mask, list(pids), list(npatches)


# =============================================================================
# Model
# =============================================================================

class DrugABMIL(nn.Module):
    """ABMIL with multi-task drug heads for drug sensitivity prediction.

    Architecture:
      encoder(input_dim → hidden_dim) → attention → weighted sum → drug_heads
    """

    def __init__(self, input_dim, hidden_dim=256, attention_dim=128,
                 n_drugs=427, dropout=0.25):
        super().__init__()

        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Attention mechanism
        self.attention_V = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Sigmoid(),
        )
        self.attention_weights = nn.Linear(attention_dim, 1)

        # Multi-task drug regression heads
        self.drug_heads = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_drugs),
        )

    def forward(self, x, mask=None, return_attention=False):
        """
        Args:
            x:    (batch, n_patches, input_dim)
            mask:  (batch, n_patches)  1=valid, 0=padding
        Returns:
            pred:      (batch, n_drugs)
            attention:  (batch, n_patches)  if return_attention
        """
        h = self.encoder(x)  # (B, N, hidden)

        # Gated attention
        a_v = self.attention_V(h)
        a_u = self.attention_U(h)
        a = self.attention_weights(a_v * a_u)  # (B, N, 1)

        if mask is not None:
            a = a.masked_fill(mask.unsqueeze(-1) == 0, -1e9)

        a = F.softmax(a, dim=1)  # (B, N, 1)
        z = torch.sum(a * h, dim=1)  # (B, hidden)

        pred = self.drug_heads(z)  # (B, n_drugs)

        if return_attention:
            return pred, a.squeeze(-1)
        return pred


# =============================================================================
# Training
# =============================================================================

def train_abmil(model, train_loader, val_loader, device, args,
                drug_mask_train=None, drug_mask_val=None):
    """Train ABMIL with multi-task drug prediction loss.

    Uses masked MSE loss (ignores NaN drug values per patient).
    Includes early stopping on validation loss.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    best_val_loss = float('inf')
    best_state = None
    patience = args.patience
    no_improve = 0

    for epoch in range(args.epochs):
        model.train()
        train_losses = []

        for batch in train_loader:
            padded, mask, drug_vals, d_mask, pids, npatches = batch
            padded = padded.to(device)
            mask = mask.to(device)
            drug_vals = drug_vals.to(device)
            d_mask = d_mask.to(device)

            pred = model(padded, mask)  # (B, n_drugs)

            # Masked MSE
            loss_mask = d_mask.bool()
            if loss_mask.sum() == 0:
                continue
            loss = F.mse_loss(pred[loss_mask], drug_vals[loss_mask])
            train_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                padded, mask, drug_vals, d_mask, pids, npatches = batch
                padded = padded.to(device)
                mask = mask.to(device)
                drug_vals = drug_vals.to(device)
                d_mask = d_mask.to(device)

                pred = model(padded, mask)
                loss_mask = d_mask.bool()
                if loss_mask.sum() == 0:
                    continue
                loss = F.mse_loss(pred[loss_mask], drug_vals[loss_mask])
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses) if train_losses else 0
        val_loss = np.mean(val_losses) if val_losses else 0

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{args.epochs}: "
                  f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# =============================================================================
# Feature Loading
# =============================================================================

def load_patch_features(features_dir, cohort):
    """Load patch-level features (no aggregation)."""
    candidates = [
        os.path.join(features_dir, f'{cohort}.cps_feature.csv'),
        os.path.join(features_dir, f'{cohort}1.cps_feature.csv'),
        os.path.join(features_dir, f'{cohort}2.cps_feature.csv'),
        os.path.join(features_dir, f'{cohort}.fm_feature.csv'),
    ]
    df = None
    for fp in candidates:
        if os.path.exists(fp):
            _df = pd.read_csv(fp)
            df = _df if df is None else pd.concat([df, _df], ignore_index=True)
    if df is None:
        return None
    # Normalize patient IDs
    df['samplename'] = df['samplename'].astype(str).str[:12].str.replace('-', '.')
    return df


def build_patch_dict(feat_df, common_patients):
    """Build {patient_id: np.array(n_patches, dim)} dict."""
    feature_cols = [c for c in feat_df.columns if c not in ('samplename', 'imgname')]
    patch_dict = {}
    for pid in common_patients:
        sub = feat_df[feat_df['samplename'] == pid]
        if len(sub) > 0:
            patch_dict[pid] = sub[feature_cols].values
    return patch_dict


# =============================================================================
# Evaluation: Per-Drug SVR
# =============================================================================

def evaluate_svr(embeddings, patient_ids, drug_df, args):
    """Per-drug SVR evaluation with 5-fold CV.

    Args:
        embeddings:  np.array (n_patients, hidden_dim)
        patient_ids: list of patient IDs
        drug_df:     DataFrame, index=patient_id, columns=drugs
        args:        CLI args

    Returns:
        results:     list of dicts with per-drug SCC
    """
    emb_df = pd.DataFrame(embeddings, index=patient_ids)
    common = set(emb_df.index) & set(drug_df.index)
    common = sorted(common)

    X_base = emb_df.loc[common].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_base)

    drug_cols = [c for c in drug_df.columns if drug_df[c].notna().sum() > 30]
    results = []

    for drug in tqdm(drug_cols, desc="Per-drug SVR", disable=len(drug_cols) < 5):
        y_all = drug_df.loc[common, drug].values.astype(float)
        valid = np.isfinite(y_all)
        if valid.sum() < 20:
            continue

        X_v = X_scaled[valid]
        y_v = y_all[valid]

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_sccs = []

        for train_idx, test_idx in kf.split(X_v):
            try:
                svr = SVR(kernel='rbf', C=1.0)
                svr.fit(X_v[train_idx], y_v[train_idx])
                pred = svr.predict(X_v[test_idx])
                scc, _ = spearmanr(y_v[test_idx], pred)
                if np.isfinite(scc):
                    fold_sccs.append(scc)
            except Exception:
                continue

        if fold_sccs:
            results.append({
                'drug': drug,
                'mean_scc': round(np.mean(fold_sccs), 4),
                'std_scc': round(np.std(fold_sccs), 4),
                'n_samples': int(valid.sum()),
                'n_folds': len(fold_sccs),
            })

    return results


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Drug Sensitivity with ABMIL")
    parser.add_argument('--model_name', type=str, required=True,
                        help='Model name (e.g., CPSformer_v2, UNI2)')
    parser.add_argument('--features_dir', type=str, required=True,
                        help='Directory with feature CSVs')
    parser.add_argument('--feature_dim', type=int, default=1024,
                        help='Feature dimension')
    parser.add_argument('--drug_csv', type=str,
                        default='/export/home/kongyan/project/cellformer/data/drug_sence_2/drug.csv')
    parser.add_argument('--output_dir', type=str, default='./results_drug_abmil')
    parser.add_argument('--cohort', type=str, default='BRCA')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--aggregation', type=str, default='abmil',
                        choices=['abmil', 'mean', 'both'],
                        help='Aggregation method')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--attention_dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--max_patches', type=int, default=100)
    parser.add_argument('--hidden_dim_abmil', type=int, default=256,
                        help='Output embedding dim from ABMIL (used for SVR)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"{'='*60}")
    print(f"Drug Sensitivity: {args.model_name} | {args.aggregation}")
    print(f"Features: {args.features_dir}")
    print(f"Device: {device}")
    print(f"{'='*60}")

    # ---- Load features ----
    print("\n[1] Loading patch features...")
    feat_df = load_patch_features(args.features_dir, args.cohort)
    if feat_df is None:
        print(f"  No features found for {args.cohort}")
        return
    n_patches = len(feat_df)
    n_patients = feat_df['samplename'].nunique()
    print(f"  Loaded {n_patches} patches from {n_patients} patients")

    # ---- Load drug data ----
    print("\n[2] Loading drug data...")
    drug_df = pd.read_csv(args.drug_csv)
    drug_df.iloc[:, 0] = drug_df.iloc[:, 0].astype(str).str[:12].str.replace('-', '.')
    drug_df = drug_df.set_index(drug_df.columns[0])
    drug_cols = [c for c in drug_df.columns if drug_df[c].notna().sum() > 30]
    print(f"  {len(drug_df)} patients, {len(drug_cols)} drugs with >30 samples")

    # ---- Common patients ----
    common = sorted(set(feat_df['samplename']) & set(drug_df.index))
    print(f"  Common patients: {len(common)}")
    if len(common) < 20:
        print("  Too few common patients, aborting")
        return

    # ---- Build patch dict ----
    print("\n[3] Building patch dictionary...")
    patch_dict = build_patch_dict(feat_df, common)
    print(f"  Patients with patches: {len(patch_dict)}")

    # ---- Drug values for common patients ----
    drug_common = drug_df.loc[common].copy()
    drug_common = drug_common.loc[:, drug_cols]

    all_results = {}

    # =====================================================================
    # Mean Pooling Baseline
    # =====================================================================
    if args.aggregation in ('mean', 'both'):
        print("\n" + "="*60)
        print("MEAN POOLING + SVR")
        print("="*60)

        feature_cols = [c for c in feat_df.columns if c not in ('samplename', 'imgname')]
        agg_df = feat_df.groupby('samplename')[feature_cols].mean()
        agg_df.index = agg_df.index.str.replace('-', '.')
        agg_common = agg_df.loc[[p for p in common if p in agg_df.index]]

        scaler = StandardScaler()
        X_mean = scaler.fit_transform(agg_common.values)

        # Per-drug SVR
        mean_results = evaluate_svr(
            X_mean, list(agg_common.index), drug_df, args)

        if mean_results:
            mean_sccs = [r['mean_scc'] for r in mean_results]
            top10 = sorted(mean_sccs, reverse=True)[:10]
            print(f"\n  Mean SCC (all drugs): {np.mean(mean_sccs):.4f}")
            print(f"  Median SCC: {np.median(mean_sccs):.4f}")
            print(f"  Top-10 mean SCC: {np.mean(top10):.4f}")

            for r in mean_results:
                r['Model'] = args.model_name
                r['aggregation'] = 'mean'
            all_results['mean'] = mean_results

    # =====================================================================
    # ABMIL Aggregation
    # =====================================================================
    if args.aggregation in ('abmil', 'both'):
        print("\n" + "="*60)
        print("ABMIL + SVR")
        print("="*60)

        feature_cols = [c for c in feat_df.columns if c not in ('samplename', 'imgname')]
        feat_dim = len(feature_cols)
        n_drugs = len(drug_cols)

        # Scale patch features
        print("\n[4] Scaling patch features...")
        all_feats = np.vstack([patch_dict[p] for p in common])
        feat_scaler = StandardScaler()
        feat_scaler.fit(all_feats)
        del all_feats

        # Scale drug values
        drug_scaler = StandardScaler()
        valid_drug_vals = drug_common.values.copy()
        drug_scaler.fit(np.nan_to_num(valid_drug_vals, nan=0.0))

        # Split patients into train/val for ABMIL training
        np.random.seed(42)
        patient_indices = np.arange(len(common))
        np.random.shuffle(patient_indices)
        n_train = int(len(common) * 0.8)
        train_pids = [common[i] for i in patient_indices[:n_train]]
        val_pids = [common[i] for i in patient_indices[n_train:]]

        # Build datasets
        train_dataset = DrugABMILDataset(
            patch_dict, drug_common, train_pids,
            feature_scaler=feat_scaler, max_patches=args.max_patches,
            drug_scaler=drug_scaler)
        val_dataset = DrugABMILDataset(
            patch_dict, drug_common, val_pids,
            feature_scaler=feat_scaler, max_patches=args.max_patches,
            drug_scaler=drug_scaler)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                 shuffle=True, collate_fn=drug_collate_fn,
                                 num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                               shuffle=False, collate_fn=drug_collate_fn,
                               num_workers=4, pin_memory=True)

        print(f"  Train patients: {len(train_dataset)}, "
              f"Val patients: {len(val_dataset)}")
        print(f"  Feature dim: {feat_dim}, Drugs: {n_drugs}")

        # ---- Train ABMIL ----
        print("\n[5] Training ABMIL...")
        model = DrugABMIL(
            input_dim=feat_dim,
            hidden_dim=args.hidden_dim,
            attention_dim=args.attention_dim,
            n_drugs=n_drugs,
            dropout=args.dropout,
        ).to(device)

        model = train_abmil(model, train_loader, val_loader, device, args)

        # ---- Extract embeddings for ALL patients ----
        print("\n[6] Extracting patient embeddings...")
        all_dataset = DrugABMILDataset(
            patch_dict, drug_common, common,
            feature_scaler=feat_scaler, max_patches=args.max_patches,
            drug_scaler=drug_scaler)
        all_loader = DataLoader(all_dataset, batch_size=args.batch_size,
                               shuffle=False, collate_fn=drug_collate_fn,
                               num_workers=4, pin_memory=True)

        model.eval()
        all_embeddings = []
        all_pids_ordered = []

        with torch.no_grad():
            for batch in all_loader:
                padded, mask, drug_vals, d_mask, pids, npatches = batch
                padded = padded.to(device)
                mask = mask.to(device)

                # Forward through encoder + attention only (get embeddings)
                h = model.encoder(padded)
                a_v = model.attention_V(h)
                a_u = model.attention_U(h)
                a = model.attention_weights(a_v * a_u)
                if mask is not None:
                    a = a.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
                a = F.softmax(a, dim=1)
                z = torch.sum(a * h, dim=1)  # (B, hidden_dim)

                all_embeddings.append(z.cpu().numpy())
                all_pids_ordered.extend(pids)

        embeddings = np.vstack(all_embeddings)  # (n_patients, hidden_dim)
        print(f"  Extracted embeddings: {embeddings.shape}")

        # ---- Per-drug SVR ----
        print("\n[7] Running per-drug SVR on ABMIL embeddings...")
        abmil_results = evaluate_svr(embeddings, all_pids_ordered, drug_df, args)

        if abmil_results:
            abmil_sccs = [r['mean_scc'] for r in abmil_results]
            top10 = sorted(abmil_sccs, reverse=True)[:10]
            print(f"\n  Mean SCC (all drugs): {np.mean(abmil_sccs):.4f}")
            print(f"  Median SCC: {np.median(abmil_sccs):.4f}")
            print(f"  Top-10 mean SCC: {np.mean(top10):.4f}")

            for r in abmil_results:
                r['Model'] = args.model_name
                r['aggregation'] = 'abmil'
            all_results['abmil'] = abmil_results

    # =====================================================================
    # Save Results
    # =====================================================================
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)

    for agg_name, results in all_results.items():
        out_csv = os.path.join(args.output_dir,
                               f'{args.cohort}_drug_{agg_name}_{args.model_name}.csv')
        pd.DataFrame(results).to_csv(out_csv, index=False)
        print(f"  Saved: {out_csv}")

    # Summary comparison
    if 'mean' in all_results and 'abmil' in all_results:
        mean_sccs = [r['mean_scc'] for r in all_results['mean']]
        abmil_sccs = [r['mean_scc'] for r in all_results['abmil']]
        print(f"\n  === {args.model_name} Summary ===")
        print(f"  Mean pooling:  SCC = {np.mean(mean_sccs):.4f}")
        print(f"  ABMIL:         SCC = {np.mean(abmil_sccs):.4f}")
        diff = np.mean(abmil_sccs) - np.mean(mean_sccs)
        print(f"  Improvement:   ΔSCC = {diff:+.4f} ({diff/np.mean(mean_sccs)*100:+.1f}%)")

    print(f"\nDone! Results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
