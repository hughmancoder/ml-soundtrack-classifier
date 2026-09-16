"""
Soundtrack Classifier CNN Training Script

Usage:
    # Run standard training from scratch:
    make train-cnn
    # (or directly): PYTHONPATH=src python scripts/train_cnn.py

    # Resume training from an existing model checkpoint:
    PYTHONPATH=src python scripts/train_cnn.py --resume models/resnet18_cnn.pt

Notes:
- The script automatically splits data into an 80/20 train/validation set.
- Class weights are dynamically computed to handle dataset imbalances.
- SpecAugment (frequency & time masking) is applied conditionally to minority classes on the GPU.
- Best model checkpoints are saved automatically when validation accuracy improves.
"""
import os
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Subset
from nnAudio.features.cqt import CQT1992v2
from soundtrack_classifier.cnn.dataset import SoundtrackCNNDataset
from soundtrack_classifier.cnn.resnet import ResNet50Classifier
import csv

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def stratified_train_val_split(
    labels: list[int],
    val_ratio: float = 0.2,
    seed: int = 42
) -> tuple[list[int], list[int]]:
    """
    Performs a stratified train/validation split ensuring every class with >= 2
    examples has representation in both training and validation subsets.
    """
    rng = np.random.RandomState(seed)
    class_indices: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        class_indices.setdefault(label, []).append(idx)

    train_indices: list[int] = []
    val_indices: list[int] = []

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


def train(resume_checkpoint=None, epochs_override=None, batch_size_override=None, lr_override=None):
    # 1. Load Configuration
    config = load_config()
    train_labels = config.get('train_labels', [
        "erhu", "pipa", "dizi", "percussion", "strings_western", "sheng",
        "suona", "guzheng", "yangqin", "woodwind", "brass", "nonmusic"
    ])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(train_labels)}
    
    device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_name)
    print(f"Training ResNet-50 CNN on device: {device}")

    # Extract CNN configuration with fallbacks to root keys for backwards compatibility
    cnn_cfg = config.get('cnn', {})
    audio_cfg = cnn_cfg.get('audio', config.get('audio', {}))
    training_cfg = cnn_cfg.get('training', config.get('training', {}))
    aug_cfg = cnn_cfg.get('augmentation', config.get('augmentation', {}))
    degradation_cfg = cnn_cfg.get('degradation', config.get('degradation', None))
    paths_cfg = cnn_cfg.get('paths', config.get('paths', {}))

    sr = audio_cfg.get('sample_rate', 44100)
    duration_sec = audio_cfg.get('duration_sec', 4.0)

    # 2. Dataset Setup
    data_dir = paths_cfg.get('data_dir', config.get('paths', {}).get('data_dir', 'data/train'))
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} not found. Please run data preparation first.")
        return

    dataset = SoundtrackCNNDataset(
        data_dir=data_dir, 
        class_to_idx=class_to_idx,
        sr=sr,
        duration_sec=duration_sec,
        degradation_config=degradation_cfg
    )
    
    # Stratified split: guarantees minority classes are represented in both train and val
    train_indices, val_indices = stratified_train_val_split(dataset.labels, val_ratio=0.2, seed=42)
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    print(f"Dataset split: {len(train_dataset)} train samples, {len(val_dataset)} validation samples.")
    
    # Compute class weights to address class imbalance (only from training set)
    train_labels_list = [dataset.labels[i] for i in train_indices]
    class_counts = np.bincount(train_labels_list, minlength=len(train_labels))
    total_train_samples = len(train_labels_list)
    
    class_counts = np.maximum(class_counts, 1) # avoid division by zero
    weights = total_train_samples / (len(train_labels) * class_counts)
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    
    # Identify minority classes (those with fewer samples than the median)
    median_count = np.median(class_counts)
    is_minority_mask = torch.tensor([1 if count < median_count else 0 for count in class_counts], dtype=torch.bool, device=device)

    batch_size = batch_size_override or training_cfg.get('batch_size', 16)
    num_workers = training_cfg.get('num_workers', 4)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    # Initialize SpecAugment transforms
    freq_mask = T.FrequencyMasking(freq_mask_param=15).to(device)
    time_mask = T.TimeMasking(time_mask_param=30).to(device)

    # 3. Model & Feature Extractor Setup
    cqt_cfg = audio_cfg.get('cqt', {})
    fmin = float(cqt_cfg.get('fmin', 65.4))
    n_bins = int(cqt_cfg.get('n_bins', 84))
    bins_per_octave = int(cqt_cfg.get('bins_per_octave', 12))

    # Initialize nnAudio CQT to run dynamically on the GPU for the entire batch
    cqt_extractor = CQT1992v2(
        sr=sr, 
        fmin=fmin, 
        n_bins=n_bins, 
        bins_per_octave=bins_per_octave
    ).to(device)

    lr = lr_override or training_cfg.get('learning_rate', 0.0001)
    model = ResNet50Classifier(num_classes=len(train_labels), pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    
    start_epoch = 0
    best_val_acc = 0.0
    epochs_no_improve = 0
    
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Resuming training from {resume_checkpoint}...")
        checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
            if 'best_val_acc' in checkpoint:
                best_val_acc = checkpoint['best_val_acc']
        else:
            model.load_state_dict(checkpoint)

    # 4. Training Loop
    epochs = epochs_override or training_cfg.get('epochs', 50)
    patience = training_cfg.get('early_stopping_patience', 10)
    
    models_dir = paths_cfg.get('models_dir', config.get('paths', {}).get('models_dir', 'models'))
    save_path = os.path.join(models_dir, "resnet50_cnn.pt")
    best_save_path = os.path.join(models_dir, "resnet50_cnn_best.pt")
    metrics_path = os.path.join(models_dir, "training_metrics.csv")
    os.makedirs(models_dir, exist_ok=True)
    
    # Initialize CSV metrics file (Append if resuming, overwrite if fresh start)
    is_resuming = resume_checkpoint and os.path.exists(resume_checkpoint)
    mode = 'a' if is_resuming and os.path.exists(metrics_path) else 'w'
    with open(metrics_path, mode=mode, newline='') as f:
        writer = csv.writer(f)
        if mode == 'w':
            writer.writerow(["epoch", "train_loss", "val_loss", "val_acc"])
    
    for epoch in range(start_epoch, epochs):
        # 4a. Training Phase
        model.train()
        running_loss = 0.0
        
        for batch_idx, (audio_batch, labels) in enumerate(train_loader):
            # Move raw audio and labels to GPU
            audio_batch = audio_batch.to(device)
            labels = labels.to(device)

            # Compute CQT dynamically on the GPU
            with torch.no_grad():
                cqt_batch = cqt_extractor(audio_batch) # output: [batch_size, freq_bins, time_steps]
                cqt_batch = cqt_batch.unsqueeze(1) # Add channel dim -> [batch, 1, freq, time]
                
                # Apply SpecAugment for robust training
                if aug_cfg.get('spec_augment', False):
                    if aug_cfg.get('minority_only', True):
                        minority_idx = is_minority_mask[labels]
                        if minority_idx.any():
                            minority_cqt = cqt_batch[minority_idx]
                            minority_cqt = freq_mask(minority_cqt)
                            minority_cqt = time_mask(minority_cqt)
                            cqt_batch[minority_idx] = minority_cqt
                    else:
                        cqt_batch = freq_mask(cqt_batch)
                        cqt_batch = time_mask(cqt_batch)

            optimizer.zero_grad()
            outputs = model(cqt_batch)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx} | Train Loss: {loss.item():.4f}")
                
        avg_train_loss = running_loss / len(train_loader)
        
        # 4b. Validation Phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for audio_batch, labels in val_loader:
                audio_batch = audio_batch.to(device)
                labels = labels.to(device)
                
                # Compute CQT dynamically
                cqt_batch = cqt_extractor(audio_batch)
                cqt_batch = cqt_batch.unsqueeze(1)
                
                # NOTE: We do NOT apply SpecAugment during validation
                
                outputs = model(cqt_batch)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100.0 * correct / total
        
        print(f"--- Epoch {epoch+1} Complete ---")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save metrics to CSV
        with open(metrics_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_train_loss, avg_val_loss, val_acc])

        # Save Best Model Checkpoint
        if val_acc > best_val_acc:
            print(f"Validation accuracy improved ({best_val_acc:.2f}% -> {val_acc:.2f}%). Saving checkpoint...")
            best_val_acc = val_acc
            epochs_no_improve = 0
            checkpoint = {
                'state_dict': model.state_dict(),
                'classes': train_labels,
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_acc': best_val_acc
            }
            torch.save(checkpoint, save_path)
            torch.save(checkpoint, best_save_path)
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation accuracy. Early stopping counter: {epochs_no_improve}/{patience}")
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered! Validation accuracy hasn't improved for {patience} epochs.")
                break
            
    print(f"Training complete! Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Best model saved to {save_path} and {best_save_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Soundtrack Classifier CNN")
    parser.add_argument("--resume", type=str, default=None, help="Path to a .pt checkpoint to resume training from")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    args = parser.parse_args()
    
    train(
        resume_checkpoint=args.resume,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
        lr_override=args.lr
    )
