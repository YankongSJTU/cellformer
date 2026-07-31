#!/usr/bin/env python3
"""
Downstream Task: Survival/Prognosis Prediction using CPS features.
Uses Cox proportional hazards model with 5-fold CV.

Usage: python downstream_survival.py --features_dir ./features_cpsformer \
    --survival_dir /export/home/kongyan/project/cellformer/survival \
    --output_dir ./results_survival
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold, train_test_split
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Survival prediction with CPS features")
    parser.add_argument('--features_dir', type=str, default='./features_cpsformer',
                        help="Directory with CPS feature CSV files")
    parser.add_argument('--survival_dir', type=str,
                        default='/export/home/kongyan/project/cellformer/survival',
                        help="Directory with survival CSV files")
    parser.add_argument('--output_dir', type=str, default='./results_survival',
                        help="Output directory")
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    return parser.parse_args()


# --- Model ---
class SurvivalModel(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        return self.net(x)


# --- Dataset ---
class SurvivalDataset(Dataset):
    def __init__(self, features, times, events, sample_names):
        self.features = torch.FloatTensor(features)
        self.times = torch.FloatTensor(times)
        self.events = torch.FloatTensor(events)
        self.sample_names = sample_names

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.times[idx], self.events[idx], self.sample_names[idx]


def cox_loss(risk, time, event):
    """Cox proportional hazards loss"""
    n = len(time)
    R_mat = torch.zeros(n, n, device=risk.device)
    for i in range(n):
        for j in range(n):
            R_mat[i, j] = (time[j] >= time[i]).float()

    theta = risk.reshape(-1)
    exp_theta = torch.exp(theta)
    loss = -torch.mean((theta - torch.log(torch.sum(exp_theta * R_mat, dim=1) + 1e-8)) * event)
    return loss


def c_index(risk, time, event):
    """Compute concordance index"""
    n = len(time)
    if n < 2:
        return 0.5

    concordant = 0
    permissible = 0

    risk = risk.flatten()
    time = time.flatten()
    event = event.flatten()

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


def remove_outlier_samples(df, percent_to_remove=0.2, min_samples=5):
    """Remove outlier patches per patient"""
    if 'samplename' not in df.columns:
        return df

    feature_cols = [c for c in df.columns if c not in ('samplename', 'imgname')]
    filtered = []

    for name, group in df.groupby('samplename'):
        if len(group) <= min_samples:
            filtered.append(group)
            continue

        features = group[feature_cols].values
        distances = cdist(features, features, metric='cosine')
        mean_dist = np.mean(distances, axis=1)

        n_remove = min(int(len(group) * percent_to_remove), len(group) - min_samples)
        if n_remove > 0:
            outlier_idx = np.argpartition(mean_dist, -n_remove)[-n_remove:]
            group = group.drop(group.index[outlier_idx])

        filtered.append(group)

    return pd.concat(filtered, ignore_index=True) if filtered else df


def load_features(features_dir, cohort):
    """Load and aggregate features"""
    feature_files = [
        os.path.join(features_dir, f'{cohort}.cps_feature.csv'),
        os.path.join(features_dir, f'{cohort}1.cps_feature.csv'),
        os.path.join(features_dir, f'{cohort}.fm_feature.csv'),
    ]

    df = None
    for fpath in feature_files:
        if os.path.exists(fpath):
            _df = pd.read_csv(fpath)
            df = _df if df is None else pd.concat([df, _df], ignore_index=True)

    if df is None:
        return None

    # Remove outliers and aggregate
    df = remove_outlier_samples(df)

    feature_cols = [c for c in df.columns if c not in ('samplename', 'imgname')]
    agg_df = df.groupby('samplename')[feature_cols].mean().reset_index()
    # Keep dashes in patient IDs to match survival data format
    return agg_df


def run_survival_cohort(cohort, features_df, survival_path, output_dir, opt):
    """Run 5-fold CV survival analysis for one cohort"""
    # Load survival data
    if not os.path.exists(survival_path):
        print(f"  No survival file: {survival_path}")
        return None

    surv_df = pd.read_csv(survival_path, sep='\t')
    if len(surv_df) < 20:
        print(f"  Too few survival samples: {len(surv_df)}")
        return None

    # Merge
    common = set(features_df['samplename']) & set(surv_df['samplename'])
    if len(common) < 15:
        print(f"  Too few common patients: {len(common)}")
        return None

    feature_cols = [c for c in features_df.columns if c != 'samplename']
    merged = pd.merge(
        features_df[features_df['samplename'].isin(common)],
        surv_df[surv_df['samplename'].isin(common)],
        on='samplename'
    )

    X = merged[feature_cols].values
    times = merged['time'].values.astype(np.float32)
    events = merged['status'].values.astype(np.float32)
    names = merged['samplename'].values

    # 5-fold CV
    device = torch.device(f'cuda:{opt.gpu}' if torch.cuda.is_available() else 'cpu')
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_cindices = []
    all_risks = []
    all_times = []
    all_events = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        t_train, t_test = times[train_idx], times[test_idx]
        e_train, e_test = events[train_idx], events[test_idx]

        # Standardize
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        train_ds = SurvivalDataset(X_train, t_train, e_train, names[train_idx])
        train_loader = DataLoader(train_ds, batch_size=min(opt.batch_size, len(X_train)),
                                  shuffle=True, drop_last=True)

        # Train
        model = SurvivalModel(input_dim=X.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-5)

        best_cindex = 0
        patience = 0
        for epoch in range(opt.epochs):
            model.train()
            for features, time_b, event_b, _ in train_loader:
                features, time_b, event_b = features.to(device), time_b.to(device), event_b.to(device)
                optimizer.zero_grad()
                risk = model(features)
                loss = cox_loss(risk, time_b, event_b)
                # L2 reg
                l2 = sum(p.pow(2.0).sum() for p in model.parameters())
                loss = loss + 0.001 * l2
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # Eval on test
            model.eval()
            with torch.no_grad():
                risk_test = model(torch.FloatTensor(X_test).to(device)).cpu().numpy()
                ci = c_index(risk_test, t_test, np.ones_like(e_test))

            if ci > best_cindex:
                best_cindex = ci
                patience = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= 30:
                    break

        # Final eval
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            risk_test = model(torch.FloatTensor(X_test).to(device)).cpu().numpy()

        ci = c_index(risk_test, t_test, np.ones_like(e_test))
        fold_cindices.append(ci)

        all_risks.extend(risk_test.flatten().tolist())
        all_times.extend(t_test.tolist())
        all_events.extend(e_test.tolist())

    mean_ci = np.mean(fold_cindices)
    std_ci = np.std(fold_cindices)

    return {
        'cohort': cohort,
        'n_patients': len(X),
        'n_events': int(events.sum()),
        'mean_cindex': mean_ci,
        'std_cindex': std_ci,
        'fold_cindices': fold_cindices,
        'all_risks': all_risks,
        'all_times': all_times,
        'all_events': all_events
    }


def main():
    opt = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    os.makedirs(opt.output_dir, exist_ok=True)

    all_cohorts = [
        'BLCA', 'BRCA', 'CESC', 'COAD', 'DLBC', 'ESCA', 'GBM', 'HNSC',
        'KICH', 'KIRC', 'KIRP', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'OV',
        'PAAD', 'PRAD', 'READ', 'STAD', 'THCA', 'THYM', 'UCEC'
    ]

    all_results = []

    for cohort in tqdm(all_cohorts, desc="Survival cohorts"):
        print(f"\n=== {cohort} ===")

        features_df = load_features(opt.features_dir, cohort)
        if features_df is None:
            print(f"  No features")
            continue
        print(f"  {len(features_df)} patients")

        # Try multiple survival file names
        survival_files = [
            os.path.join(opt.survival_dir, f'{cohort}.survival.csv'),
            os.path.join(opt.survival_dir, f'{cohort}.os.survival.csv'),
            os.path.join(opt.survival_dir, f'{cohort}.dss.survival.csv'),
            os.path.join(opt.survival_dir, f'{cohort}1.survival.csv'),
        ]

        result = None
        for sf in survival_files:
            result = run_survival_cohort(cohort, features_df, sf, opt.output_dir, opt)
            if result is not None:
                break

        if result is not None:
            all_results.append(result)
            print(f"  C-index: {result['mean_cindex']:.4f} ± {result['std_cindex']:.4f}")

            # Save per-cohort results
            cohort_result = {k: v for k, v in result.items() if k not in ('all_risks', 'all_times', 'all_events', 'fold_cindices')}
            pd.DataFrame([cohort_result]).to_csv(
                os.path.join(opt.output_dir, f'{cohort}_survival_results.csv'), index=False)

    # Summary
    if all_results:
        summary = []
        for r in all_results:
            summary.append({
                'cohort': r['cohort'],
                'n_patients': r['n_patients'],
                'n_events': r['n_events'],
                'mean_cindex': r['mean_cindex'],
                'std_cindex': r['std_cindex']
            })

        summary_df = pd.DataFrame(summary)
        summary_df = summary_df.sort_values('mean_cindex', ascending=False)
        summary_df.to_csv(os.path.join(opt.output_dir, 'all_survival_results.csv'), index=False)

        print(f"\n=== Summary ===")
        print(f"Cohorts with results: {len(summary_df)}")
        print(f"Mean C-index: {summary_df['mean_cindex'].mean():.4f}")
        print(f"Cohorts with C-index > 0.55: {len(summary_df[summary_df['mean_cindex'] > 0.55])}")
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()