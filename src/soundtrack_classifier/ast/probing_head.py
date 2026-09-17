from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, TensorDataset


class ProbingHead(nn.Module):
    """
    Two-layer Multi-Layer Perceptron (MLP) Probing Head for Frozen Foundation Embeddings.
    Maps fused 1,280-d embeddings (MERT 768-d + CLAP 512-d) to multi-label class logits.

    Architecture:
        1,280 -> Linear -> LayerNorm -> GELU -> Dropout(0.4)
              -> 512    -> Linear -> LayerNorm -> GELU -> Dropout(0.3)
              -> 256    -> Linear -> 12 (Logits)
    """
    def __init__(
        self,
        in_dim: int = 1280,
        hidden_dims: Optional[List[int]] = None,
        num_classes: int = 12,
        dropouts: Optional[List[float]] = None,
        use_layer_norm: bool = True
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]
        if dropouts is None:
            dropouts = [0.4, 0.3]

        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes

        layers: List[nn.Module] = []
        current_dim = in_dim

        for i, (h_dim, p_drop) in enumerate(zip(hidden_dims, dropouts)):
            layers.append(nn.Linear(current_dim, h_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.GELU())
            if p_drop > 0:
                layers.append(nn.Dropout(p_drop))
            current_dim = h_dim

        layers.append(nn.Linear(current_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input embeddings of shape [Batch, in_dim]
        Returns:
            logits: Output logits of shape [Batch, num_classes]
        """
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input embeddings of shape [Batch, in_dim]
        Returns:
            probs: Sigmoid probability distributions of shape [Batch, num_classes]
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)


class ProbingHeadTrainer:
    """
    5-Fold GroupKFold Trainer for Probing Head.
    Stratifies / groups samples by original file recording ID to avoid data leakage
    between sliced time windows.
    """
    def __init__(
        self,
        in_dim: int = 1280,
        hidden_dims: Optional[List[int]] = None,
        num_classes: int = 12,
        dropouts: Optional[List[float]] = None,
        lr: float = 0.001,
        weight_decay: float = 0.01,
        epochs: int = 40,
        batch_size: int = 64,
        n_splits: int = 5,
        early_stopping_patience: int = 10,
        device: Optional[str] = None
    ):
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims or [512, 256]
        self.num_classes = num_classes
        self.dropouts = dropouts or [0.4, 0.3]
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_splits = n_splits
        self.early_stopping_patience = early_stopping_patience

        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

    def _compute_pos_weights(self, y_train: np.ndarray) -> torch.Tensor:
        """
        Calculates positive class weight ratio (negatives / positives)
        for weighted BCEWithLogitsLoss.
        """
        num_samples = y_train.shape[0]
        pos_counts = np.sum(y_train, axis=0)
        pos_counts = np.maximum(pos_counts, 1.0)
        neg_counts = num_samples - pos_counts
        weights = neg_counts / pos_counts
        # Clamp extreme values for numerical stability
        weights = np.clip(weights, 1.0, 50.0)
        return torch.tensor(weights, dtype=torch.float32, device=self.device)

    def train_single_split(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray,
        Y_val: np.ndarray,
        fold: int = 1
    ) -> Tuple[ProbingHead, Dict[str, float]]:
        """
        Trains a single model on train split and validates on val split.
        """
        model = ProbingHead(
            in_dim=self.in_dim,
            hidden_dims=self.hidden_dims,
            num_classes=self.num_classes,
            dropouts=self.dropouts
        ).to(self.device)

        pos_weights = self._compute_pos_weights(Y_train)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-6)

        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(Y_train, dtype=torch.float32)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(Y_val, dtype=torch.float32)
        )

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        best_val_loss = float("inf")
        best_f1_macro = 0.0
        best_state = None
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(batch_x)
            scheduler.step()

            # Evaluation
            model.eval()
            val_loss = 0.0
            all_preds: List[np.ndarray] = []
            all_targets: List[np.ndarray] = []

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    logits = model(batch_x)
                    loss = criterion(logits, batch_y)
                    val_loss += loss.item() * len(batch_x)

                    probs = torch.sigmoid(logits).cpu().numpy()
                    all_preds.append(probs)
                    all_targets.append(batch_y.cpu().numpy())

            val_loss /= len(val_dataset)
            y_pred_probs = np.vstack(all_preds)
            y_true = np.vstack(all_targets)
            y_pred_bin = (y_pred_probs >= 0.5).astype(int)

            f1_macro = f1_score(y_true, y_pred_bin, average="macro", zero_division=0)
            f1_micro = f1_score(y_true, y_pred_bin, average="micro", zero_division=0)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_f1_macro = f1_macro
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.early_stopping_patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        metrics = {
            "val_loss": float(best_val_loss),
            "f1_macro": float(best_f1_macro),
            "f1_micro": float(f1_micro)
        }
        return model, metrics

    def train_kfold(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        groups: np.ndarray,
        save_checkpoint_path: Optional[str] = "models/ast_probing_head.pt"
    ) -> Tuple[ProbingHead, List[Dict[str, float]]]:
        """
        Executes 5-fold GroupKFold training, printing summary statistics and saving
        the overall best performing checkpoint.
        """
        gkf = GroupKFold(n_splits=self.n_splits)
        fold_metrics: List[Dict[str, float]] = []
        best_overall_f1 = -1.0
        best_overall_model: Optional[ProbingHead] = None

        print(f"Starting {self.n_splits}-fold GroupKFold training on device: {self.device}")
        print(f"Total samples: {len(X)}, Unique recording files/groups: {len(np.unique(groups))}")

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, Y, groups=groups), start=1):
            X_train, Y_train = X[train_idx], Y[train_idx]
            X_val, Y_val = X[val_idx], Y[val_idx]

            model, metrics = self.train_single_split(
                X_train, Y_train, X_val, Y_val, fold=fold
            )
            fold_metrics.append(metrics)
            print(f"Fold {fold}/{self.n_splits} - Val Loss: {metrics['val_loss']:.4f} | F1-Macro: {metrics['f1_macro']:.4f} | F1-Micro: {metrics['f1_micro']:.4f}")

            if metrics["f1_macro"] > best_overall_f1:
                best_overall_f1 = metrics["f1_macro"]
                best_overall_model = model

        avg_loss = np.mean([m["val_loss"] for m in fold_metrics])
        avg_f1 = np.mean([m["f1_macro"] for m in fold_metrics])
        print(f"\n--- GroupKFold Summary: Mean Val Loss = {avg_loss:.4f}, Mean F1-Macro = {avg_f1:.4f} ---")

        if save_checkpoint_path and best_overall_model is not None:
            Path(save_checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": best_overall_model.state_dict(),
                "in_dim": self.in_dim,
                "hidden_dims": self.hidden_dims,
                "num_classes": self.num_classes,
                "dropouts": self.dropouts,
                "fold_metrics": fold_metrics,
                "best_f1_macro": best_overall_f1
            }, save_checkpoint_path)
            print(f"Saved best probing head checkpoint to: {save_checkpoint_path}")

        return best_overall_model or model, fold_metrics
