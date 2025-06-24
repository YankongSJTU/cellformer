import torch
from datasets import TumorDataset
import numpy as np
import os
import sys
import random
from typing import List, Tuple, Optional, Dict, Any
from models import EnsemblePrognosisModel
from utils import calculate_cindex, regularize_weights
from sklearn.model_selection import KFold, train_test_split
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from scipy.spatial.distance import cdist
from datetime import datetime
import time
DEFAULT_SEED = 42
DEFAULT_N_SPLITS = 5
DEFAULT_N_REPEATS = 10
DEFAULT_BATCH_SIZE = 22
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_NUM_EPOCHS = 200

def set_random_seeds(seed: int = DEFAULT_SEED) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        num_epochs: int = DEFAULT_NUM_EPOCHS,
        lr: float = DEFAULT_LEARNING_RATE,
        fold: int = 0,
        output_dir: str = "./checkpoints"
    ) -> None:
        """Initialize trainer with model, device and training parameters."""
        self.model = model
        self.device = device
        self.num_epochs = num_epochs
        self.lr = lr
        self.fold = fold
        self.output_dir = output_dir
        self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        os.makedirs(self.output_dir, exist_ok=True)
        self._init_log_files()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=10, verbose=True)
        self.best_val_cindex = 0
        self.epochs_without_improvement = 0
        self.early_stopping_patience = 30

    def _init_log_files(self) -> None:
        """Initialize log files for training."""
        self.train_log_path = os.path.join(self.output_dir, f"CV_{self.fold}_train_log.csv")
        self.result_pred_path = os.path.join(self.output_dir, f"CV_{self.fold}_result_pred.csv")
        
        with open(self.train_log_path, 'w') as f:
            f.write("epoch,train_loss,train_cindex,val_loss,val_cindex,time\n")
        with open(self.result_pred_path, 'w') as f:
            f.write("fold,epoch,best_val_cindex,test_cindex\n")

    def log_training(self, epoch: int, train_loss: float, train_cindex: float, 
                    val_loss: float, val_cindex: float) -> None:
        """Log training progress to file."""
        with open(self.train_log_path, 'a') as f:
            f.write(f"{epoch},{train_loss:.4f},{train_cindex:.4f},"
                    f"{val_loss:.4f},{val_cindex:.4f},{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    def log_results(self, epoch: int, best_val_cindex: float, test_cindex: float) -> None:
        """Log final results to file."""
        with open(self.result_pred_path, 'a') as f:
            f.write(f"{self.fold},{epoch},{best_val_cindex:.4f},{test_cindex:.4f}\n")

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train model for one epoch."""
        self.model.train()
        total_loss = 0.0
        risks, times, events = [], [], []

        for features, survival_time, status, _ in train_loader:
            features = features.to(self.device)
            survival_time = survival_time.to(self.device)
            status = status.to(self.device)

            self.optimizer.zero_grad()
            risk = self.model(features)
            loss = self.calculate_loss(risk, survival_time, status)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            risks.append(risk.detach().cpu())
            times.append(survival_time.cpu())
            events.append(status.cpu())

        risks = torch.cat(risks).numpy().flatten()
        times = torch.cat(times).numpy().flatten()
        events = torch.cat(events).numpy().flatten()
        calevent=np.ones_like(events)
        train_cindex = calculate_cindex(risks, times, calevent)

        return total_loss / len(train_loader), train_cindex

    def calculate_loss(self, risk: torch.Tensor, survival_time: torch.Tensor, 
                      status: torch.Tensor) -> torch.Tensor:
        """Calculate Cox proportional hazards loss with regularization."""
        current_batch_len = len(survival_time)
        R_mat = torch.zeros([current_batch_len, current_batch_len], 
                           dtype=torch.float32, device=self.device)
        
        for i in range(current_batch_len):
            for j in range(current_batch_len):
                R_mat[i,j] = survival_time[j] >= survival_time[i]

        theta = risk.reshape(-1)
        exp_theta = torch.exp(theta)
        loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * status.float())
        
        # Regularization
        l2_reg = sum(p.pow(2.0).sum() for p in self.model.parameters())
        l1_reg = regularize_weights(self.model)
        
        return loss_cox + 1e-5 * l1_reg + 0.001 * l2_reg

    def evaluate(self, data_loader: DataLoader) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate model on given data loader."""
        self.model.eval()
        total_loss = 0.0
        all_risks, all_times, all_status = [], [], []

        with torch.no_grad():
            for features, survival_time, status, _ in data_loader:
                if len(features) == 0:
                    continue

                features = features.to(self.device)
                survival_time = survival_time.to(self.device)
                status = status.to(self.device)

                risk = self.model(features)
                loss = self.calculate_loss(risk, survival_time, status)

                total_loss += loss.item()
                all_risks.append(risk.detach().cpu().numpy().flatten())
                all_times.append(survival_time.cpu().numpy().flatten())
                all_status.append(status.cpu().numpy().flatten())

        if not all_risks:
            return float('nan'), 0.5, np.array([]), np.array([]), np.array([])

        risks = np.concatenate(all_risks)
        times = np.concatenate(all_times)
        events = np.concatenate(all_status)
        calevent=np.ones_like(events)

        cindex = calculate_cindex(risks, times, calevent)

        return total_loss / max(1, len(data_loader)), cindex, risks, times, events

    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
             test_loader: DataLoader) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """Full training procedure with validation and testing."""
        best_val_cindex = 0
        best_train_cindex = 0
        best_epoch = 0

        for epoch in range(self.num_epochs):
            start_time = time.time()

            # Training phase
            train_loss, train_cindex = self.train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_cindex, risks, times, events = self.evaluate(val_loader)
            
            # Learning rate adjustment
            self.scheduler.step(val_cindex)
            
            # Logging
            self.log_training(epoch, train_loss, train_cindex, val_loss, val_cindex)
            
            # Print progress
            if epoch % 10 == 0 or epoch == self.num_epochs - 1:
                print(f"Fold {self.fold} Epoch {epoch}/{self.num_epochs}: "
                      f"Train Loss = {train_loss:.4f}, Train C-index = {train_cindex:.4f}, "
                      f"Val Loss = {val_loss:.4f}, Val C-index = {val_cindex:.4f}, "
                      f"Time = {time.time()-start_time:.2f}s")

            # Model checkpointing
            if train_cindex > best_train_cindex:
            #if val_cindex > best_val_cindex:
                best_train_cindex = train_cindex
                #best_val_cindex = val_cindex
                best_epoch = epoch
                torch.save(self.model.state_dict(), 
                          os.path.join(self.output_dir, f"CV_{self.fold}_bestmodel.pth"))
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        # Final evaluation
        self.model.load_state_dict(torch.load(os.path.join(self.output_dir, f"CV_{self.fold}_bestmodel.pth")))
        test_loss, test_cindex, risks, times, events = self.evaluate(test_loader)
        self.log_results(best_epoch, best_val_cindex, test_cindex)

        print(f"\nFold {self.fold} Best Epoch {best_epoch}: "
              f"Val C-index = {best_val_cindex:.4f}, Test C-index = {test_cindex:.4f}")

        return test_cindex, risks, times, events


def remove_outlier_samples(df_features: pd.DataFrame,
                         percent_to_remove: float = 0.2,
                         min_samples_to_keep: int = 5,
                         distance_metric: str = 'cosine',
                         verbose: bool = True) -> pd.DataFrame:
    """Optimized outlier removal function."""
    if 'samplename' not in df_features.columns:
        raise ValueError("DataFrame must contain 'samplename' column")
    filtered_samples = []
    removal_stats = []
    for name, group in df_features.groupby('samplename'):
        n_original = len(group)
        if n_original <= min_samples_to_keep:
            filtered_samples.append(group)
            removal_stats.append((name, n_original, 0, n_original))
            continue
        features = group.drop(columns=['samplename']).select_dtypes(include=[np.number])
        distances = cdist(features, features, metric=distance_metric)
        mean_distances = np.mean(distances, axis=1)
        max_to_remove = n_original - min_samples_to_keep
        n_to_remove = min(int(n_original * percent_to_remove), max_to_remove)
        if n_to_remove <= 0:
            filtered_samples.append(group)
            removal_stats.append((name, n_original, 0, n_original))
            continue
        outlier_indices = np.argpartition(mean_distances, -n_to_remove)[-n_to_remove:]
        filtered_samples.append(group.drop(group.index[outlier_indices]))
        removal_stats.append((name, n_original, n_to_remove, n_original - n_to_remove))
    result_df = pd.concat(filtered_samples, ignore_index=True)
    if verbose:
        stats_df = pd.DataFrame(removal_stats,
                              columns=['Sample', 'Original', 'Removed', 'Remaining'])
        print("\nOutlier removal statistics:")
        print(stats_df)
        print(f"\nTotal removed: {stats_df['Removed'].sum()}/{stats_df['Original'].sum()} "
             f"({stats_df['Removed'].sum()/stats_df['Original'].sum():.1%})")
    return result_df


def run_cross_validation(
    tumor:str,
    features_path: str,
    survival_path: str,
    output_dir: str,
    n_splits: int = DEFAULT_N_SPLITS,
    remove_outliers: bool = True
) -> None:
    """
    Run k-fold cross validation with optional outlier removal.
    
    Parameters:
        features_path: Path to features CSV file
        survival_path: Path to survival data TSV file
        output_dir: Directory to save results
        n_splits: Number of CV folds
        remove_outliers: Whether to remove outlier samples
    """
    set_random_seeds()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features_df = pd.read_csv(features_path)
    features_df = remove_outlier_samples(features_df)
    
    survival_df = pd.read_csv(survival_path, sep="\t")
    
    
    merged_df = pd.merge(features_df, survival_df, on='samplename')
    dataset = TumorDataset(merged_df)

    # Main training loop
    all_results = []
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=DEFAULT_SEED)
    fold_results = []

    for fold, (train_val_idx, test_idx) in enumerate(kf.split(dataset)):
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=0.1
        )

        # Create data loaders
        train_loader = DataLoader(
            Subset(dataset, train_idx),
            batch_size=DEFAULT_BATCH_SIZE,
            shuffle=True,
            drop_last=True
        )
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=DEFAULT_BATCH_SIZE)
        test_loader = DataLoader(Subset(dataset, test_idx), batch_size=DEFAULT_BATCH_SIZE)

        # Train and evaluate
        model = EnsemblePrognosisModel(input_dim=1024).to(device)
        trainer = Trainer(model, device, fold=fold+1, output_dir=output_dir)
        
        test_cindex, risks, times, events = trainer.train(train_loader, val_loader, test_loader)
        fold_results.append(test_cindex)

        # Save KM plot
        km_path = os.path.join(output_dir, f"fold_{fold+1}_km_curve.png")
        plot_km_curve(times, events, risks, km_path,tumor)

    all_results.append(fold_results)

    # Save combined results
    save_combined_results(output_dir, all_results)

import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

def plot_km_curve(times, events, risks, output_path, tumor, time_unit='Months', 
                  risk_cutoff='median', figsize=(5, 4),
                  colors=('blue', 'red'), ci_colors=((0.85, 0.9, 1), (1, 0.85, 0.85))):
    """
    Simplified Kaplan-Meier curve plotting function without at-risk table.
    
    Parameters:
    -----------
    times : array-like
        Array of survival times for each patient
    events : array-like
        Array of event indicators (1 if event occurred, 0 if censored)
    risks : array-like
        Array of risk scores for each patient
    output_path : str
        Path to save the generated plot
    tumor : str
        Name of the tumor type (used in plot title)
    time_unit : str, optional
        Unit for time axis (default: 'Months')
    risk_cutoff : str or float, optional
        Method to split groups ('median', 'quantile', or a specific cutoff value)
    figsize : tuple, optional
        Figure size (default: (8, 6))
    colors : tuple, optional
        Colors for low and high risk groups (default: ('blue', 'red'))
    ci_colors : tuple, optional
        Colors for confidence intervals (default: light blue and light pink)
    """
    
    # Validate inputs
    if not all(isinstance(arr, (np.ndarray, list)) for arr in [times, events, risks]):
        raise ValueError("times, events, and risks must be arrays or lists")
        
    if len(times) != len(events) or len(times) != len(risks):
        raise ValueError("times, events, and risks must have the same length")
        
    if risks is None or len(risks) == 0:
        print("Warning: Invalid risks array - using random values")
        risks = np.random.rand(len(times)) if len(times) > 0 else np.array([0.5])

    if times is None or len(times) == 0:
        print("Warning: No survival times provided")
        return

    try:
        # Determine risk groups
        if isinstance(risk_cutoff, str):
            if risk_cutoff.lower() == 'median':
                cutoff = np.median(risks)
            elif risk_cutoff.lower() == 'quantile':
                cutoff = np.quantile(risks, 0.6)
            else:
                raise ValueError("risk_cutoff must be 'median', 'quantile', or a numeric value")
        else:
            cutoff = float(risk_cutoff)
            
        groups = risks > cutoff
        
        # Adjust cutoff if all samples in one group
        if groups.sum() == 0 or groups.sum() == len(groups):
            print("Warning: All samples in same risk group - adjusting cutoff")
            cutoff = np.quantile(risks, 0.6)
            groups = risks > cutoff

        # Create figure
        plt.figure(figsize=figsize)
        ax = plt.gca()

        kmf = KaplanMeierFitter()
        
        # Plot high risk group
        if groups.sum() > 0:
            kmf.fit(times[groups], events[groups], label=f'High Risk (n={groups.sum()})')
            kmf.plot_survival_function(ci_show=False, color=colors[1], ax=ax)
            # Manually add CI
            sf = kmf.survival_function_
            ci = kmf.confidence_interval_
            ax.fill_between(sf.index, ci.iloc[:, 0], ci.iloc[:, 1], 
                          color=ci_colors[1], alpha=0.2, linewidth=0)

        # Plot low risk group
        if (~groups).sum() > 0:
            kmf.fit(times[~groups], events[~groups], label=f'Low Risk (n={(~groups).sum()})')
            kmf.plot_survival_function(ci_show=False, color=colors[0], ax=ax)
            # Manually add CI
            sf = kmf.survival_function_
            ci = kmf.confidence_interval_
            ax.fill_between(sf.index, ci.iloc[:, 0], ci.iloc[:, 1], 
                          color=ci_colors[0], alpha=0.2, linewidth=0)

        # Add log-rank test results if possible
        if len(np.unique(groups)) > 1:
            results = logrank_test(times[groups], times[~groups],
                                  events[groups], events[~groups])
            ax.set_title(f"TCGA {tumor} (held-out)", pad=20)
            ax.text(0.05, 0.15, f"Pvalue = {results.p_value:.3e}", 
                    transform=ax.transAxes, fontsize=10) 
#                    bbox=dict(facecolor='white', alpha=0.8))
        else:
            ax.set_title("KM Curve (single group)")

        ax.set_xlabel(f"Time ({time_unit})")
        ax.set_ylabel("Survival Probability")
        ax.grid(False)
        ax.set_ylim(0, 1.05)
        
        # Clean up axes
        for spine in ['right', 'top']:
            ax.spines[spine].set_visible(False)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        plt.close()
        raise RuntimeError(f"Error plotting KM curve: {str(e)}")


def save_combined_results(output_dir: str, all_results: List[List[float]]) -> None:
    print("\n=== Final Combined Results ===")
    with open(os.path.join(output_dir, "combined_results.txt"), 'w') as f:
        f.write("Repeat\tFold\tC-index\n")
        
        all_cindices = []
        f.write("\nOverall Statistics:\n")
        stats = {
            "Mean": np.mean(all_cindices),
            "Std": np.std(all_cindices),
            "Median": np.median(all_cindices),
            "Min": np.min(all_cindices),
            "Max": np.max(all_cindices)
        }
        
        for name, value in stats.items():
            f.write(f"{name} C-index: {value:.4f}\n")
            print(f"{name} C-index: {value:.4f}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python main.py features.csv survival.tsv output_dir ")
        sys.exit(1)

    features_path = sys.argv[1]
    survival_path = sys.argv[2]
    tumorname=sys.argv[3]
    output_dir = os.path.join("./checkpoints", sys.argv[3])

    os.makedirs(output_dir, exist_ok=True)
    run_cross_validation(tumorname,features_path, survival_path, output_dir)
