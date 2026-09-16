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

import os
import argparse
import csv
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from soundtrack_classifier.rnn.dataset import ArchivalSoundtrackRNNDataset
from soundtrack_classifier.rnn.rnn import ArchivalInstrumentCRNN


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_rnn(
    resume_checkpoint=None,
    epochs_override=None,
    batch_size_override=None,
    sr=22050,
    crop_duration_sec=8.0
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
    data_dir = config["paths"]["data_dir"]
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} not found.")
        return

    print(f"Loading dataset with dynamic {crop_duration_sec}s crops at {sr} Hz...")
    full_dataset = ArchivalSoundtrackRNNDataset(
        data_dir=data_dir,
        class_to_idx=class_to_idx,
        sr=sr,
        crop_duration_sec=crop_duration_sec,
        random_crop=True,
        degradation_config=config.get("degradation", None)
    )

    if len(full_dataset) == 0:
        print(f"Error: No audio files found in {data_dir}.")
        return

    # Train/Val split: 80% train, 20% validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )

    # Compute class weights to counteract class imbalance
    train_labels_list = [full_dataset.labels[i] for i in train_dataset.indices]
    class_counts = np.bincount(train_labels_list, minlength=num_classes)
    class_counts = np.maximum(class_counts, 1)
    weights = len(train_labels_list) / (num_classes * class_counts)
    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

    batch_size = batch_size_override or config.get("training", {}).get("batch_size", 16)
    num_workers = min(4, config.get("training", {}).get("num_workers", 2))

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

    learning_rate = config.get("training", {}).get("learning_rate", 0.0003)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

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
    epochs = epochs_override or config.get("training", {}).get("epochs", 40)
    patience = config.get("training", {}).get("early_stopping_patience", 7)

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

        with torch.no_grad():
            for audio_batch, labels_batch in val_loader:
                audio_batch = audio_batch.to(device)
                labels_batch = labels_batch.to(device)

                outputs = model(audio_batch, return_frames=False)
                loss = criterion(outputs, labels_batch)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels_batch.size(0)
                correct += (predicted == labels_batch).sum().item()

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
    args = parser.parse_args()

    train_rnn(
        resume_checkpoint=args.resume,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
        sr=args.sr,
        crop_duration_sec=args.crop_duration
    )
