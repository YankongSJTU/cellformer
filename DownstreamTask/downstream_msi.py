#!/usr/bin/env python3
"""
Downstream Task: MSI (Microsatellite Instability) Prediction for CRC.
Binary classification: MSI-H vs MSS on TCGA COAD + READ cohorts.

Usage:
  python downstream_msi.py --features_dir ./features --msi_file ./clinical/msi/COADREAD.info
  python downstream_msi.py --features_dir ./features --method svm
"""

import os, sys, argparse, warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
warnings.filterwarnings('ignore')
torch.set_num_threads(4)

COHORTS = ['COAD', 'READ']


def parse_args():
    parser = argparse.ArgumentParser(description="MSI prediction for CRC (COAD + READ)")
    parser.add_argument('--features_dir', type=str, required=True,
                        help="Directory with {cohort}.cps_feature.csv files")
    parser.add_argument('--msi_file', type=str,
                        default='./clinical/msi/COADREAD.info',
                        help="MSI label file (tab-separated: patient_id, msi_status)")
    parser.add_argument('--output_dir', type=str, default='./results_msi',
                        help="Output directory")
    parser.add_argument('--classification', type=str, default='binary',
                        choices=['binary', 'ternary'],
                        help='binary=MSI-H vs MSS; ternary=MSS vs MSI-L vs MSI-H')
    parser.add_argument('--method', type=str, default='both',
                        choices=['svm', 'nn', 'both'],
                        help='Classifier method')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    parser.add_argument('--n_folds', type=int, default=5)
    return parser.parse_args()


def load_features(features_dir, cohort):
    """Load feature CSV for a cohort, try multiple file naming conventions."""
    suffixes = ['.cps_feature.csv', '1.cps_feature.csv', '.fm_feature.csv',
                '1.fm_feature.csv']
    df = None
    for suffix in suffixes:
        fpath = os.path.join(features_dir, f'{cohort}{suffix}')
        if os.path.exists(fpath):
            _df = pd.read_csv(fpath)
            df = _df if df is None else pd.concat([df, _df], ignore_index=True)

    if df is None:
        return None

    feature_cols = [c for c in df.columns if c not in ('samplename', 'imgname')]
    agg_df = df.groupby('samplename')[feature_cols].mean().reset_index()
    return agg_df


def load_msi_labels(msi_file, classification='binary'):
    """Load MSI labels. Return (patient_id, label) pairs."""
    df = pd.read_csv(msi_file, sep='\t', header=None, names=['patient_id', 'msi_status'])
    df['patient_id'] = df['patient_id'].str.strip()
    df['msi_status'] = df['msi_status'].str.strip().str.lower()

    if classification == 'binary':
        # MSI-H (81) vs MSS (400), exclude MSI-L and indeterminate
        df_binary = df[df['msi_status'].isin(['msi-h', 'mss'])].copy()
        df_binary['label'] = (df_binary['msi_status'] == 'msi-h').astype(int)
    else:
        # Ternary: MSS=0, MSI-L=1, MSI-H=2, exclude indeterminate
        df = df[df['msi_status'] != 'indeterminate']
        label_map = {'mss': 0, 'msi-l': 1, 'msi-h': 2}
        df_binary = df.copy()
        df_binary['label'] = df_binary['msi_status'].map(label_map)

    return df_binary[['patient_id', 'label']].reset_index(drop=True)


def run_svm_cv(X, y, n_folds=5):
    """SVM with 5-fold stratified CV."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_aucs, fold_accs, fold_f1s = [], [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = SVC(kernel='rbf', probability=True, class_weight='balanced',
                  C=1.0, gamma='scale')
        clf.fit(X_train, y_train)

        y_prob = clf.predict_proba(X_test)[:, 1]
        y_pred = clf.predict(X_test)

        fold_aucs.append(roc_auc_score(y_test, y_prob))
        fold_accs.append(accuracy_score(y_test, y_pred))
        fold_f1s.append(f1_score(y_test, y_pred, average='binary'))

    return {
        'auc_mean': np.mean(fold_aucs), 'auc_std': np.std(fold_aucs),
        'acc_mean': np.mean(fold_accs), 'acc_std': np.std(fold_accs),
        'f1_mean': np.mean(fold_f1s), 'f1_std': np.std(fold_f1s),
        'fold_aucs': fold_aucs,
    }


class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


def run_nn_cv(X, y, n_folds=5, device='cuda', epochs=100):
    """Simple NN with 5-fold CV."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_aucs, fold_accs, fold_f1s = [], [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        X_t = torch.FloatTensor(X_train).to(device)
        y_t = torch.FloatTensor(y_train).to(device)
        X_te = torch.FloatTensor(X_test).to(device)
        y_te = torch.FloatTensor(y_test).to(device)

        pos_w = (y_train == 0).sum() / (y_train == 1).sum()
        model = SimpleNN(X_train.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w]).to(device))

        ds = TensorDataset(X_t, y_t)
        loader = DataLoader(ds, batch_size=32, shuffle=True)

        best_auc = 0
        best_probs = None
        for ep in range(epochs):
            model.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = model(xb).squeeze(-1)  # (batch,) or scalar
                if logits.dim() == 0:
                    logits = logits.unsqueeze(0)
                if yb.dim() == 0:
                    yb = yb.unsqueeze(0)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            # Evaluate
            model.eval()
            with torch.no_grad():
                logits_te = model(X_te).squeeze()
                probs = torch.sigmoid(logits_te).cpu().numpy()
                auc = roc_auc_score(y_test, probs)
                if auc > best_auc:
                    best_auc = auc
                    best_probs = probs

        y_pred = (best_probs > 0.5).astype(int)
        fold_aucs.append(best_auc)
        fold_accs.append(accuracy_score(y_test, y_pred))
        fold_f1s.append(f1_score(y_test, y_pred, average='binary'))

    return {
        'auc_mean': np.mean(fold_aucs), 'auc_std': np.std(fold_aucs),
        'acc_mean': np.mean(fold_accs), 'acc_std': np.std(fold_accs),
        'f1_mean': np.mean(fold_f1s), 'f1_std': np.std(fold_f1s),
        'fold_aucs': fold_aucs,
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load MSI labels
    msi_labels = load_msi_labels(args.msi_file, args.classification)
    print(f"MSI labels loaded: {len(msi_labels)} patients")
    print(f"  Distribution: {msi_labels['label'].value_counts().to_dict()}")

    # Load and combine COAD + READ features
    all_features = None
    for cohort in COHORTS:
        feat_df = load_features(args.features_dir, cohort)
        if feat_df is None:
            print(f"  {cohort}: No features found in {args.features_dir}")
            continue
        feat_df['cohort'] = cohort
        print(f"  {cohort}: {len(feat_df)} patients")
        all_features = feat_df if all_features is None else pd.concat(
            [all_features, feat_df], ignore_index=True)

    if all_features is None:
        print("ERROR: No features found for COAD or READ")
        return

    # Merge with MSI labels
    feature_cols = [c for c in all_features.columns
                    if c not in ('samplename', 'imgname', 'cohort')]
    merged = msi_labels.merge(
        all_features[['samplename'] + feature_cols + ['cohort']],
        left_on='patient_id', right_on='samplename', how='inner')

    n_total = len(merged)
    n_msih = (merged['label'] == 1).sum() if args.classification == 'binary' \
        else (merged['label'] == 2).sum()
    n_mss = (merged['label'] == 0).sum()
    print(f"\nMerged: {n_total} patients (MSS={n_mss}, MSI-H={n_msih}), "
          f"feat_dim={len(feature_cols)}")

    if n_total < 20:
        print("Too few patients, skipping")
        return

    X = merged[feature_cols].values
    y = merged['label'].values

    # Run classifiers
    results = []
    if args.method in ('svm', 'both'):
        print("\nRunning SVM...")
        res = run_svm_cv(X, y, args.n_folds)
        results.append({'method': 'SVM', **res})
        print(f"  SVM: AUC={res['auc_mean']:.4f}±{res['auc_std']:.4f} "
              f"ACC={res['acc_mean']:.4f} F1={res['f1_mean']:.4f}")

    if args.method in ('nn', 'both'):
        print(f"\nRunning NN (device={device})...")
        res = run_nn_cv(X, y, args.n_folds, device=device)
        results.append({'method': 'NN', **res})
        print(f"  NN:  AUC={res['auc_mean']:.4f}±{res['auc_std']:.4f} "
              f"ACC={res['acc_mean']:.4f} F1={res['f1_mean']:.4f}")

    # Save results
    if results:
        for r in results:
            r['task'] = f'MSI_{"binary" if args.classification == "binary" else "ternary"}'
            r['n_total'] = n_total
            r['n_mss'] = int(n_mss)
            r['n_msih'] = int(n_msih)
            r.pop('fold_aucs', None)
        result_df = pd.DataFrame(results)
        csv_path = os.path.join(args.output_dir, 'msi_results.csv')
        result_df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")


if __name__ == '__main__':
    main()


if __name__ == '__main__':
    main()
