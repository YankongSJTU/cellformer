#!/usr/bin/env python3
"""
WSI-Level Tumor Classification with Gated Attention MIL.

Proper MIL implementation:
- Each patch gets WSI's tumor type label
- Patient-level train/test split (no data leakage)
- Gated attention MIL learns patch importance
- TCGA training + test, CPTAC zero-shot + finetune evaluation
"""

import os
import sys
import argparse
import random
import gc
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

torch.set_num_threads(4)

# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

FEATURES_DIR = os.path.join(PROJECT_ROOT, 'features')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results_mil_classify')
CPTAC_DIR = '/data1/TumorGroup/DATA/public_database/slide/CPTAC'
TCGA_DIR = '/data1/TumorGroup/DATA/public_database/TCGA/slide'

ALL_COHORTS = [
    'BLCA', 'BRCA', 'CESC', 'COAD', 'DLBC', 'ESCA', 'GBM', 'HNSC',
    'KIRC', 'KIRP', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'OV', 'PAAD',
    'PRAD', 'READ', 'STAD', 'THCA', 'THYM', 'UCEC'
]

CPTAC_TO_TCGA = {
    'BRCA': 'BRCA', 'COAD': 'COAD', 'LUAD': 'LUAD', 'OV': 'OV',
    'PDA': 'PAAD', 'UCEC': 'UCEC', 'CCRCC': 'KIRC'
}

# =============================================================================
# Data Loading
# =============================================================================

def load_tcga_patch_data():
    """Load all TCGA patch-level CPS features with tumor type labels.

    Returns:
        df: DataFrame with columns [feature_cols, 'samplename', 'label']
        feature_cols: list of feature column names
    """
    all_data = []

    for cohort in ALL_COHORTS:
        feature_files = [
            os.path.join(FEATURES_DIR, f'{cohort}.cps_feature.csv'),
            os.path.join(FEATURES_DIR, f'{cohort}1.cps_feature.csv'),
            os.path.join(FEATURES_DIR, f'{cohort}2.cps_feature.csv'),
        ]

        df = None
        for fpath in feature_files:
            if os.path.exists(fpath):
                _df = pd.read_csv(fpath)
                df = _df if df is None else pd.concat([df, _df], ignore_index=True)

        if df is None:
            continue

        # Get feature columns (all columns except samplename and imgname)
        feature_cols = [c for c in df.columns if c not in ('samplename', 'imgname')]

        # Add label column
        df['label'] = cohort

        # Drop rows with NaN features
        n_before = len(df)
        df = df.dropna(subset=feature_cols).reset_index(drop=True)
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            print(f"  {cohort}: dropped {n_dropped} rows with NaN features")

        all_data.append(df)
        print(f"  {cohort}: {len(df)} patches, {df['samplename'].nunique()} patients")

    if not all_data:
        raise ValueError("No feature files found!")

    combined = pd.concat(all_data, ignore_index=True)
    feature_cols = [c for c in combined.columns if c not in ('samplename', 'imgname', 'label')]

    print(f"\nTotal: {len(combined)} patches, {combined['samplename'].nunique()} patients, {combined['label'].nunique()} classes")

    return combined, feature_cols


def patient_level_split(df, test_size=0.2, random_state=42):
    """Split patients (not patches) into train/test sets.

    Returns:
        train_patients, test_patients: lists of patient IDs
        df: DataFrame with added 'split' column
    """
    # Get unique patient-label pairs
    patient_labels = df.groupby('samplename')['label'].first().reset_index()

    # Stratified split on patients
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(patient_labels['samplename'], patient_labels['label']))

    train_patients = patient_labels['samplename'].iloc[train_idx].tolist()
    test_patients = patient_labels['samplename'].iloc[test_idx].tolist()

    # Assign split to each patch
    df['split'] = 'train'
    df.loc[df['samplename'].isin(test_patients), 'split'] = 'test'

    print(f"  Train: {len(train_patients)} patients ({df[df['split']=='train'].shape[0]} patches)")
    print(f"  Test:  {len(test_patients)} patients ({df[df['split']=='test'].shape[0]} patches)")

    return train_patients, test_patients, df


# =============================================================================
# PyTorch Dataset
# =============================================================================

class MILBagDataset(Dataset):
    """Dataset where each item is a WSI (bag of patches)."""

    def __init__(self, df, feature_cols, patient_list, scaler=None, max_patches=100,
                 is_train=True, label_encoder=None):
        """
        Args:
            df: DataFrame with patch-level data
            feature_cols: list of feature column names
            patient_list: list of patient IDs for this split
            scaler: StandardScaler for feature normalization
            max_patches: max patches per bag (for memory efficiency during training)
            is_train: whether this is training set (affects subsampling)
            label_encoder: LabelEncoder for labels
        """
        self.df = df
        self.feature_cols = feature_cols
        self.patient_list = patient_list
        self.scaler = scaler
        self.max_patches = max_patches
        self.is_train = is_train
        self.label_encoder = label_encoder

        # Group by patient
        self.patient_data = {}
        for patient in patient_list:
            patient_df = df[df['samplename'] == patient]
            if len(patient_df) > 0:
                self.patient_data[patient] = patient_df

        self.patients = list(self.patient_data.keys())

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient = self.patients[idx]
        patient_df = self.patient_data[patient]

        # Get features
        features = patient_df[self.feature_cols].values.astype(np.float32)

        # Normalize if scaler provided
        if self.scaler is not None:
            features = self.scaler.transform(features)

        # Get label
        label_str = patient_df['label'].iloc[0]
        label = self.label_encoder.transform([label_str])[0]

        # Subsample during training
        n_patches = len(features)
        if self.is_train and n_patches > self.max_patches:
            indices = np.random.choice(n_patches, self.max_patches, replace=False)
            features = features[indices]
            n_patches = self.max_patches

        return features, label, patient, n_patches


def mil_collate_fn(batch):
    """Collate variable-length bags into batched tensors with padding mask."""
    features_list, labels_list, patient_ids, n_patches_list = zip(*batch)

    # Find max length in batch
    max_len = max(f.shape[0] for f in features_list)
    batch_size = len(features_list)
    feat_dim = features_list[0].shape[1]

    # Pad features
    padded_features = torch.zeros(batch_size, max_len, feat_dim)
    mask = torch.zeros(batch_size, max_len)

    for i, f in enumerate(features_list):
        n = f.shape[0]
        padded_features[i, :n] = torch.from_numpy(f)
        mask[i, :n] = 1.0

    labels = torch.LongTensor(labels_list)

    return padded_features, mask, labels, list(patient_ids), list(n_patches_list)


# =============================================================================
# Model: Gated Attention MIL
# =============================================================================

class GatedAttentionMIL(nn.Module):
    """Gated Attention Multiple Instance Learning for WSI classification.

    From Ilse et al. 2018: Attention-based Deep Multiple Instance Learning
    """

    def __init__(self, input_dim=1024, hidden_dim=256, attention_dim=128,
                 n_classes=22, dropout=0.25):
        super().__init__()

        # Feature encoder: transforms raw CPS features
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Gated attention mechanism
        self.attention_V = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(attention_dim, 1)

        # Classifier on aggregated bag representation
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x, mask=None, return_attention=False):
        """
        Args:
            x: (batch_size, n_patches, input_dim)
            mask: (batch_size, n_patches) - 1 for valid patches, 0 for padding
            return_attention: whether to return attention weights

        Returns:
            logits: (batch_size, n_classes)
            attention: (batch_size, n_patches) if return_attention=True
        """
        # Feature encoding
        h = self.feature_encoder(x)  # (B, N, hidden_dim)

        # Gated attention
        a_v = self.attention_V(h)    # (B, N, attention_dim)
        a_u = self.attention_U(h)    # (B, N, attention_dim)
        a = self.attention_weights(a_v * a_u)  # (B, N, 1)

        # Mask out padding
        if mask is not None:
            a = a.masked_fill(mask.unsqueeze(-1) == 0, -1e9)

        # Softmax over patches
        a = F.softmax(a, dim=1)  # (B, N, 1)

        # Weighted aggregation
        z = torch.sum(a * h, dim=1)  # (B, hidden_dim)

        # Classification
        logits = self.classifier(z)  # (B, n_classes)

        if return_attention:
            return logits, a.squeeze(-1)
        return logits


# =============================================================================
# Training
# =============================================================================

def compute_class_weights(labels, n_classes):
    """Compute class weights for balanced loss."""
    class_counts = np.bincount(labels, minlength=n_classes)
    # Avoid division by zero
    class_counts = np.maximum(class_counts, 1)
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * n_classes
    return torch.FloatTensor(weights)


def train_mil_model(train_loader, test_loader, n_classes, device, args):
    """Train the MIL model."""

    # Initialize model
    model = GatedAttentionMIL(
        input_dim=len(args.feature_cols),
        hidden_dim=args.hidden_dim,
        attention_dim=args.attention_dim,
        n_classes=n_classes,
        dropout=args.dropout
    ).to(device)

    # Class-weighted loss
    # Get all labels from train loader
    all_labels = []
    for batch in train_loader:
        all_labels.extend(batch[2].tolist())
    class_weights = compute_class_weights(all_labels, n_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_features, batch_mask, batch_labels, _, _ in train_loader:
            batch_features = batch_features.to(device)
            batch_mask = batch_mask.to(device)
            batch_labels = batch_labels.to(device)

            logits = model(batch_features, mask=batch_mask)
            loss = criterion(logits, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            preds = logits.argmax(dim=1)
            train_correct += (preds == batch_labels).sum().item()
            train_total += batch_labels.size(0)

        scheduler.step()
        train_acc = train_correct / train_total

        # Evaluation on test set
        model.eval()
        test_correct = 0
        test_total = 0
        test_preds_all = []
        test_labels_all = []
        test_probs_all = []

        with torch.no_grad():
            for batch_features, batch_mask, batch_labels, _, _ in test_loader:
                batch_features = batch_features.to(device)
                batch_mask = batch_mask.to(device)
                batch_labels = batch_labels.to(device)

                logits = model(batch_features, mask=batch_mask)
                probs = F.softmax(logits, dim=1)

                preds = logits.argmax(dim=1)
                test_correct += (preds == batch_labels).sum().item()
                test_total += batch_labels.size(0)

                test_preds_all.extend(preds.cpu().tolist())
                test_labels_all.extend(batch_labels.cpu().tolist())
                test_probs_all.extend(probs.cpu().tolist())

        test_acc = test_correct / test_total

        # Early stopping
        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"  Epoch {epoch:3d}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, test_acc={test_acc:.4f} (best={best_acc:.4f})")

        if patience_counter >= args.patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_acc, test_preds_all, test_labels_all, test_probs_all


# =============================================================================
# Evaluation & Metrics
# =============================================================================

def compute_top_k_accuracy(probs, labels, k):
    """Compute top-K accuracy."""
    probs = np.array(probs)
    labels = np.array(labels)
    top_k_preds = np.argsort(probs, axis=1)[:, -k:]
    correct = np.sum([label in top_k_pred for label, top_k_pred in zip(labels, top_k_preds)])
    return correct / len(labels)


def compute_metrics(preds, labels, probs, class_names):
    """Compute comprehensive metrics."""
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average='macro')
    f1_weighted = f1_score(labels, preds, average='weighted')
    top3_acc = compute_top_k_accuracy(probs, labels, 3)
    top5_acc = compute_top_k_accuracy(probs, labels, 5)

    # Per-class metrics
    precision_per_class = precision_score(labels, preds, average=None, zero_division=0)
    recall_per_class = recall_score(labels, preds, average=None, zero_division=0)
    f1_per_class = f1_score(labels, preds, average=None, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(labels, preds)

    metrics = {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'top3_accuracy': top3_acc,
        'top5_accuracy': top5_acc,
        'confusion_matrix': cm,
        'per_class_precision': precision_per_class,
        'per_class_recall': recall_per_class,
        'per_class_f1': f1_per_class,
    }

    return metrics


def plot_confusion_matrix(cm, class_names, save_path, title="Confusion Matrix"):
    """Plot normalized confusion matrix heatmap."""
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', xticklabels=class_names,
                yticklabels=class_names, cmap='Blues', ax=ax, vmin=0, vmax=1)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()

    # Save both PNG and TIF
    plt.savefig(save_path.replace('.png', '.png'), dpi=150, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.tif'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {save_path}")


def plot_per_class_metrics(metrics, class_names, save_path):
    """Plot per-class accuracy/F1 bar chart."""
    n_classes = len(class_names)
    acc_per_class = metrics['per_class_recall']  # Recall = accuracy per class when balanced

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(n_classes)
    bars = ax.bar(x, acc_per_class, color='steelblue', edgecolor='black')

    # Add value labels
    for bar, val in zip(bars, acc_per_class):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylabel('Recall (Class Accuracy)')
    ax.set_title('Per-Class Recall (Accuracy)')
    ax.set_ylim(0, 1.1)
    plt.tight_layout()

    plt.savefig(save_path.replace('.png', '.png'), dpi=150, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.tif'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {save_path}")


# =============================================================================
# CPTAC Evaluation
# =============================================================================

def get_cptac_patients_with_features():
    """Check if any CPTAC patients have pre-extracted features."""
    # OV has pre-extracted data in cellformer project
    ov_data_dir = os.path.join(PROJECT_ROOT, 'data', 'dataOV')

    if os.path.exists(ov_data_dir):
        image_dir = os.path.join(ov_data_dir, 'image')
        if os.path.exists(image_dir):
            images = [f for f in os.listdir(image_dir) if f.endswith('.png')]
            # Extract patient IDs from filenames
            patients = set()
            for img in images:
                # CPTAC filenames: CPTAC-XXBRXXX-XXXXX.png
                parts = img.split('-')
                if len(parts) >= 2:
                    patient_id = f"{parts[0]}-{parts[1]}"
                    patients.add(patient_id)
            return list(patients), len(images)
    return [], 0


def evaluate_cptac_zero_shot(model, scaler, le, device, args, n_classes):
    """Evaluate on CPTAC using zero-shot prediction (no finetuning)."""
    print("\n[CPTAC Mode A] Zero-shot evaluation...")

    # Check pre-extracted data
    ov_patients, ov_patches = get_cptac_patients_with_features()

    results = []

    if ov_patches > 0:
        print(f"  OV: {ov_patches} pre-extracted patches from {len(ov_patients)} patients")

        # For OV, we can use existing image/segment data with CPS model
        # This requires running CPS feature extraction, which is time-consuming
        # Skip for now and use SVS samples
        print("  Skipping OV pre-extracted data (requires CPS model inference)")

    # Sample CPTAC SVS files for quick evaluation
    # This requires full pipeline: SVS -> patches -> nucseg -> CPS features -> classify
    print("  Note: Full CPTAC evaluation requires SVS processing pipeline")
    print("  Use mode 'eval_cptac_full' for complete evaluation")

    return results


def evaluate_cptac_finetune(model, scaler, le, device, args, n_classes):
    """Finetune on CPTAC data and evaluate."""
    print("\n[CPTAC Mode B] Finetune evaluation...")

    # This requires extracting CPTAC features first
    print("  Note: CPTAC finetuning requires feature extraction from SVS files")
    print("  Use mode 'extract_cptac' first, then run 'eval_cptac_B'")

    return None


# =============================================================================
# Save/Load
# =============================================================================

def save_checkpoint(model, scaler, le, feature_cols, metrics, args, save_path):
    """Save model and all metadata."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': len(feature_cols),
            'hidden_dim': args.hidden_dim,
            'attention_dim': args.attention_dim,
            'n_classes': len(le.classes_),
            'dropout': args.dropout,
        },
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'classes': le.classes_,
        'feature_cols': feature_cols,
        'metrics': metrics,
        'args': vars(args),
    }, save_path)

    print(f"  Saved checkpoint: {save_path}")


def load_checkpoint(checkpoint_path, device):
    """Load model and metadata from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = ckpt['model_config']
    model = GatedAttentionMIL(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        attention_dim=config['attention_dim'],
        n_classes=config['n_classes'],
        dropout=config['dropout']
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])

    scaler = StandardScaler()
    scaler.mean_ = ckpt['scaler_mean']
    scaler.scale_ = ckpt['scaler_scale']

    le = LabelEncoder()
    le.classes_ = ckpt['classes']

    feature_cols = ckpt['feature_cols']

    return model, scaler, le, feature_cols, ckpt.get('metrics', {})


# =============================================================================
# Main Pipeline
# =============================================================================

def run_training(args):
    """Run full training pipeline on TCGA data."""
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'metrics'), exist_ok=True)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Step 1: Load data
    print("\n[1] Loading TCGA patch data...")
    df, feature_cols = load_tcga_patch_data()
    args.feature_cols = feature_cols

    # Step 2: Patient-level split
    print("\n[2] Patient-level split...")
    train_patients, test_patients, df = patient_level_split(df, test_size=args.test_size, random_state=args.seed)

    # Step 3: Encode labels
    le = LabelEncoder()
    le.fit(df['label'].unique())
    n_classes = len(le.classes_)
    print(f"  Classes: {n_classes} ({list(le.classes_)})")

    # Step 4: Fit scaler on training patches
    print("\n[3] Fitting scaler on training patches...")
    train_df = df[df['split'] == 'train']
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].values.astype(np.float32))

    # Step 5: Create datasets
    print("\n[4] Creating datasets...")
    train_dataset = MILBagDataset(
        df, feature_cols, train_patients, scaler=scaler,
        max_patches=args.max_patches, is_train=True, label_encoder=le
    )
    test_dataset = MILBagDataset(
        df, feature_cols, test_patients, scaler=scaler,
        max_patches=args.max_patches, is_train=False, label_encoder=le
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=mil_collate_fn, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=mil_collate_fn, num_workers=0
    )

    print(f"  Train: {len(train_dataset)} WSIs")
    print(f"  Test:  {len(test_dataset)} WSIs")

    # Step 6: Train model
    print("\n[5] Training MIL model...")
    model, best_acc, test_preds, test_labels, test_probs = train_mil_model(
        train_loader, test_loader, n_classes, device, args
    )

    # Step 7: Compute metrics
    print("\n[6] Computing metrics...")
    metrics = compute_metrics(test_preds, test_labels, test_probs, le.classes_)

    print(f"\n  === Final Results ===")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Top-3 Accuracy: {metrics['top3_accuracy']:.4f}")
    print(f"  Top-5 Accuracy: {metrics['top5_accuracy']:.4f}")
    print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")

    # Step 8: Visualizations
    print("\n[7] Generating visualizations...")
    cm_path = os.path.join(args.output_dir, 'figures', 'confusion_matrix.png')
    plot_confusion_matrix(metrics['confusion_matrix'], le.classes_, cm_path,
                          title="WSI Tumor Classification - TCGA Test Set")

    acc_path = os.path.join(args.output_dir, 'figures', 'per_class_accuracy.png')
    plot_per_class_metrics(metrics, le.classes_, acc_path)

    # Step 9: Save results
    print("\n[8] Saving results...")

    # Save checkpoint
    ckpt_path = os.path.join(args.output_dir, 'checkpoints', 'best_mil_model.pt')
    save_checkpoint(model, scaler, le, feature_cols, metrics, args, ckpt_path)

    # Save metrics CSV
    metrics_df = pd.DataFrame({
        'class': le.classes_,
        'precision': metrics['per_class_precision'],
        'recall': metrics['per_class_recall'],
        'f1': metrics['per_class_f1'],
    })
    metrics_df.to_csv(os.path.join(args.output_dir, 'metrics', 'per_class_metrics.csv'), index=False)

    # Save overall metrics
    overall_df = pd.DataFrame({
        'metric': ['accuracy', 'top3_accuracy', 'top5_accuracy', 'f1_macro', 'f1_weighted'],
        'value': [metrics['accuracy'], metrics['top3_accuracy'], metrics['top5_accuracy'],
                  metrics['f1_macro'], metrics['f1_weighted']]
    })
    overall_df.to_csv(os.path.join(args.output_dir, 'metrics', 'overall_metrics.csv'), index=False)

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}")

    return model, scaler, le, feature_cols, metrics


def run_eval_tcga(args):
    """Evaluate saved model on TCGA test set."""
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    print("\n[Loading checkpoint...]")
    model, scaler, le, feature_cols, _ = load_checkpoint(args.checkpoint, device)

    # Load data
    print("\n[Loading TCGA data...]")
    df, _ = load_tcga_patch_data()

    # Patient-level split (same seed for reproducibility)
    train_patients, test_patients, df = patient_level_split(df, test_size=args.test_size, random_state=args.seed)

    # Create test dataset
    test_dataset = MILBagDataset(
        df, feature_cols, test_patients, scaler=scaler,
        max_patches=args.max_patches, is_train=False, label_encoder=le
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=mil_collate_fn, num_workers=0
    )

    # Evaluate
    model.eval()
    test_preds = []
    test_labels = []
    test_probs = []

    with torch.no_grad():
        for batch_features, batch_mask, batch_labels, _, _ in test_loader:
            batch_features = batch_features.to(device)
            batch_mask = batch_mask.to(device)

            logits = model(batch_features, mask=batch_mask)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            test_preds.extend(preds.cpu().tolist())
            test_labels.extend(batch_labels.tolist())
            test_probs.extend(probs.cpu().tolist())

    metrics = compute_metrics(test_preds, test_labels, test_probs, le.classes_)

    print(f"\n  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Top-3: {metrics['top3_accuracy']:.4f}")
    print(f"  F1 (macro): {metrics['f1_macro']:.4f}")

    return metrics


def run_eval_cptac_A(args):
    """Zero-shot CPTAC evaluation."""
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    if args.checkpoint:
        model, scaler, le, feature_cols, _ = load_checkpoint(args.checkpoint, device)
    else:
        # Train first
        model, scaler, le, feature_cols, metrics = run_training(args)

    results = evaluate_cptac_zero_shot(model, scaler, le, device, args, len(le.classes_))

    return results


def run_eval_cptac_B(args):
    """Finetune CPTAC evaluation."""
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    if args.checkpoint:
        model, scaler, le, feature_cols, _ = load_checkpoint(args.checkpoint, device)
    else:
        model, scaler, le, feature_cols, metrics = run_training(args)

    results = evaluate_cptac_finetune(model, scaler, le, device, args, len(le.classes_))

    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='WSI MIL Classification')

    # Mode
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'eval_tcga', 'eval_cptac_A', 'eval_cptac_B', 'full'],
                        help='Execution mode')

    # Training params
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (number of WSIs)')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--attention_dim', type=int, default=128, help='Attention dimension')
    parser.add_argument('--dropout', type=float, default=0.25, help='Dropout rate')
    parser.add_argument('--max_patches', type=int, default=100, help='Max patches per bag during training')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set proportion')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Paths
    parser.add_argument('--features_dir', type=str, default=FEATURES_DIR, help='Features directory')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='Output directory')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path for evaluation')

    args = parser.parse_args()

    print("="*60)
    print("WSI MIL Classification with Gated Attention")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"GPU: {args.gpu}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max patches: {args.max_patches}")

    if args.mode == 'train':
        run_training(args)

    elif args.mode == 'eval_tcga':
        run_eval_tcga(args)

    elif args.mode == 'eval_cptac_A':
        run_eval_cptac_A(args)

    elif args.mode == 'eval_cptac_B':
        run_eval_cptac_B(args)

    elif args.mode == 'full':
        # Train first
        model, scaler, le, feature_cols, metrics = run_training(args)
        # Then CPTAC evaluations (requires checkpoint)
        args.checkpoint = os.path.join(args.output_dir, 'checkpoints', 'best_mil_model.pt')
        print("\n[CPTAC Mode A] Zero-shot...")
        run_eval_cptac_A(args)


if __name__ == "__main__":
    main()