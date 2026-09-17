"""
Soundtrack Classifier CRNN Training Script
Optimized for 6-to-8 second dynamic window training with 2-channel HPSS + CQT front-end.

Usage:
    # Run training from scratch:
    make train-rnn
    # or directly:
    PYTHONPATH=src python src/train_rnn.py

    # Resume training from an existing checkpoint:
    PYTHONPATH=src python src/train_rnn.py --resume models/crnn.pt
"""

import argparse
import csv
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

from soundtrack_classifier.rnn.dataset import ArchivalSoundtrackRNNDataset
from soundtrack_classifier.rnn.rnn import ArchivalInstrumentCRNN


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class audio classification.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Down-weights easy background/dominant class frames while focusing learning
    on hard, sparsely occurring instrument events.
    """
    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean"
    ):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 1. Unweighted CE to get true p_t
        ce_loss_unweighted = F.cross_entropy(
            logits, targets, reduction="none", weight=None
        )
        pt = torch.exp(-ce_loss_unweighted)
        modulating_factor = (1.0 - pt) ** self.gamma

        # 2. Weighted CE to scale gradients
        ce_loss_weighted = F.cross_entropy(
            logits, targets, reduction="none", weight=self.weight
        )
        loss = modulating_factor * ce_loss_weighted
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class BinaryFocalLossWithLogits(nn.Module):
    """Multi-label binary focal loss with logits."""
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[float] = 0.25,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = "mean"
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = bce_loss * ((1 - p_t) ** self.gamma)
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def stratified_train_val_split(
    labels: List[int],
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[List[int], List[int]]:
    """
    Performs a stratified train/validation split ensuring every class with >= 2
    examples has representation in both training and validation subsets.
    Eliminates zero-validation class dropouts on small datasets.
    """
    rng = np.random.RandomState(seed)
    class_indices: dict = {}
    for idx, label in enumerate(labels):
        class_indices.setdefault(label, []).append(idx)

    train_indices: List[int] = []
    val_indices: List[int] = []

    for _, indices in sorted(class_indices.items()):
        indices = list(indices)
        rng.shuffle(indices)
        n = len(indices)
        if n == 1:
            train_indices.extend(indices)
        elif n == 2:
            train_indices.append(indices[0])
            val_indices.append(indices[1])
        else:
            n_val = max(1, int(round(n * val_ratio)))
            val_indices.extend(indices[:n_val])
            train_indices.extend(indices[n_val:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def train_rnn(
    resume_checkpoint=None,
    epochs_override=None,
    batch_size_override=None,
    sr=22050,
    crop_duration_sec=8.0,
    features_dir_override=None,
    precompute_features_override=None
):
    # 1. Configuration & Classes
    config = load_config()
    train_labels = config.get("train_labels", [
        "sheng", "dizi", "erhu", "pipa", "suona", "guzheng",
        "yangqin", "percussion", "strings", "woodwind", "brass", "nonmusic"
    ])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(train_labels)}
    num_classes = len(train_labels)

    device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_name)
    print(f"Training Archival CRNN on device: {device}")

    # 2. Dataset Setup
    crnn_train_cfg = config.get("crnn", {}).get("training", {})
    data_dir = (
        config.get("paths", {}).get("data_dir") or
        config.get("crnn", {}).get("paths", {}).get("data_dir") or
        config.get("cnn", {}).get("paths", {}).get("data_dir", "data/train")
    )
    precompute_flag = (
        precompute_features_override
        if precompute_features_override is not None
        else crnn_train_cfg.get("precompute_features", True)
    )
    features_dir = features_dir_override or crnn_train_cfg.get("features_dir", "data/features_cqt_hpss")

    mode_desc = f"Precomputed Cache Mode ({features_dir})" if precompute_flag else "Dynamic Waveform Mode"
    print(f"Loading dataset with dynamic {crop_duration_sec}s crops at {sr} Hz [{mode_desc}]...")
    full_dataset = ArchivalSoundtrackRNNDataset(
        data_dir=data_dir,
        class_to_idx=class_to_idx,
        sr=sr,
        crop_duration_sec=crop_duration_sec,
        random_crop=True,
        degradation_config=config.get("crnn", {}).get("degradation") or config.get("degradation", None),
        features_dir=features_dir,
        precompute_features=precompute_flag
    )

    if len(full_dataset) == 0:
        print(f"Error: No audio files found in {data_dir}.")
        return

    # Train/Val split: Stratified split ensuring balanced representation across all classes
    train_indices, val_indices = stratified_train_val_split(
        labels=full_dataset.labels,
        val_ratio=0.2,
        seed=42
    )
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    print(f"[*] Stratified Split: {len(train_dataset)} train samples, {len(val_dataset)} validation samples")

    # Compute class weights (inverse frequency) for multi-class Focal Loss
    train_labels_list = [full_dataset.labels[i] for i in train_dataset.indices]
    class_counts = np.bincount(train_labels_list, minlength=num_classes)
    class_counts = np.maximum(class_counts, 1)

    total_train_samples = len(train_labels_list)
    class_weights = total_train_samples / (num_classes * class_counts)
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    crnn_train_cfg = config.get("crnn", {}).get("training", {})
    batch_size = batch_size_override or crnn_train_cfg.get("batch_size") or config.get("training", {}).get("batch_size", 16)
    num_workers = min(4, crnn_train_cfg.get("num_workers") or config.get("training", {}).get("num_workers", 4))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    # 3. Model, Loss, Optimizer
    model = ArchivalInstrumentCRNN(
        num_classes=num_classes,
        in_channels=2,
        sr=sr
    ).to(device)

    learning_rate = crnn_train_cfg.get("learning_rate") or config.get("training", {}).get("learning_rate", 0.0003)
    focal_gamma = float(crnn_train_cfg.get("focal_gamma", 2.0))
    focal_alpha = crnn_train_cfg.get("focal_alpha", None)
    if focal_alpha is not None:
        focal_alpha = float(focal_alpha)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = FocalLoss(
        gamma=focal_gamma,
        weight=weights_tensor
    )

    start_epoch = 0
    best_val_acc = 0.0
    epochs_no_improve = 0

    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Resuming CRNN training from {resume_checkpoint}...")
        checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "epoch" in checkpoint:
                start_epoch = checkpoint["epoch"] + 1
            if "best_val_acc" in checkpoint:
                best_val_acc = checkpoint["best_val_acc"]
        else:
            model.load_state_dict(checkpoint)

    # 4. Training Loop
    epochs = epochs_override or crnn_train_cfg.get("epochs") or config.get("training", {}).get("epochs", 40)
    patience = crnn_train_cfg.get("early_stopping_patience") or config.get("training", {}).get("early_stopping_patience", 7)

    models_dir = config.get("paths", {}).get("models_dir", "models")
    os.makedirs(models_dir, exist_ok=True)
    best_path = os.path.join(models_dir, "crnn_best.pt")
    last_path = os.path.join(models_dir, "crnn.pt")
    metrics_path = os.path.join(models_dir, "rnn_training_metrics.csv")

    is_resuming = resume_checkpoint and os.path.exists(resume_checkpoint)
    mode = "a" if is_resuming and os.path.exists(metrics_path) else "w"
    with open(metrics_path, mode=mode, newline="") as f:
        writer = csv.writer(f)
        if mode == "w":
            writer.writerow(["epoch", "train_loss", "val_loss", "val_acc"])

    print(f"Starting training for {epochs} epochs (Early stopping patience: {patience})...")

    for epoch in range(start_epoch, epochs):
        # Training Phase
        model.train()
        running_loss = 0.0

        for batch_idx, (audio_batch, labels_batch) in enumerate(train_loader):
            audio_batch = audio_batch.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad()
            # Forward pass: returns clip-level logits pooled over sequence
            outputs = model(audio_batch, return_frames=False)

            # 1. Targets are already class indices, pass directly to multi-class loss
            loss = criterion(outputs, labels_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 15 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")

        avg_train_loss = running_loss / max(1, len(train_loader))

        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        # 4. Dominant class selection during validation
        with torch.no_grad():
            for audio_batch, labels_batch in val_loader:
                audio_batch = audio_batch.to(device)
                labels_batch = labels_batch.to(device)

                outputs = model(audio_batch, return_frames=False)
                loss = criterion(outputs, labels_batch)
                val_loss += loss.item()

                probs = torch.softmax(outputs, dim=-1)
                predicted_dominant = torch.argmax(probs, dim=-1)
                total += labels_batch.size(0)
                correct += (predicted_dominant == labels_batch).sum().item()

        avg_val_loss = val_loss / max(1, len(val_loader))
        val_acc = 100.0 * correct / max(1, total)

        print(f"--- Epoch {epoch+1} Complete ---")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Log metrics
        with open(metrics_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_train_loss, avg_val_loss, val_acc])

        checkpoint_data = {
            "state_dict": model.state_dict(),
            "classes": train_labels,
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_acc": best_val_acc
        }
        torch.save(checkpoint_data, last_path)

        # Checkpoint Best Model
        if val_acc > best_val_acc:
            print(f"Validation accuracy improved ({best_val_acc:.2f}% -> {val_acc:.2f}%). Saving {best_path}...")
            best_val_acc = val_acc
            epochs_no_improve = 0
            checkpoint_data["best_val_acc"] = best_val_acc
            torch.save(checkpoint_data, best_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break

    print(f"\nCRNN Training complete. Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to {best_path} and {last_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Archival Instrument CRNN")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--sr", type=int, default=22050, help="Sample rate (default: 22050)")
    parser.add_argument("--crop-duration", type=float, default=8.0, help="Training crop duration in seconds (default: 8.0)")
    parser.add_argument("--features-dir", type=str, default=None, help="Directory containing precomputed CQT+HPSS features (default: data/features_cqt_hpss)")
    parser.add_argument("--precompute-features", dest="precompute_features", action="store_true", default=None, help="Force precomputed CQT+HPSS cached mode")
    parser.add_argument("--dynamic-features", dest="precompute_features", action="store_false", help="Force dynamic waveform loading and CQT extraction")
    args = parser.parse_args()

    train_rnn(
        resume_checkpoint=args.resume,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
        sr=args.sr,
        crop_duration_sec=args.crop_duration,
        features_dir_override=args.features_dir,
        precompute_features_override=args.precompute_features
    )
