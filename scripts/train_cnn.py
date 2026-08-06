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
from torch.utils.data import DataLoader, random_split
from nnAudio.features.cqt import CQT1992v2
from soundtrack_classifier.cnn.dataset import SoundtrackCNNDataset
from soundtrack_classifier.cnn.resnet import ResNet18Classifier

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def train(resume_checkpoint=None):
    # 1. Load Configuration
    config = load_config()
    train_labels = config['train_labels']
    class_to_idx = {cls_name: i for i, cls_name in enumerate(train_labels)}
    
    device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_name)
    print(f"Training MVP CNN on device: {device}")

    # 2. Dataset Setup
    data_dir = config['paths']['data_dir']
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} not found. Please run the pure slicing step first.")
        return

    dataset = SoundtrackCNNDataset(
        data_dir=data_dir, 
        class_to_idx=class_to_idx,
        sr=config['audio']['sample_rate'],
        duration_sec=config['audio']['duration_sec'],
        degradation_config=config.get('degradation', None)
    )
    
    # Split into 80% train, 20% validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    # Compute class weights to address class imbalance (only from training set)
    train_labels_list = [dataset.labels[i] for i in train_dataset.indices]
    class_counts = np.bincount(train_labels_list, minlength=len(train_labels))
    total_train_samples = len(train_labels_list)
    
    class_counts = np.maximum(class_counts, 1) # avoid division by zero
    weights = total_train_samples / (len(train_labels) * class_counts)
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    
    # Identify minority classes (those with fewer samples than the median)
    median_count = np.median(class_counts)
    is_minority_mask = torch.tensor([1 if count < median_count else 0 for count in class_counts], dtype=torch.bool, device=device)

    # We can safely use num_workers > 0 because CQT is not in __getitem__ anymore
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        num_workers=config['training']['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )

    # Initialize SpecAugment transforms
    freq_mask = T.FrequencyMasking(freq_mask_param=15).to(device)
    time_mask = T.TimeMasking(time_mask_param=30).to(device)

    # 3. Model & Feature Extractor Setup
    # Initialize nnAudio CQT to run dynamically on the GPU for the entire batch
    cqt_extractor = CQT1992v2(
        sr=config['audio']['sample_rate'], 
        fmin=config['audio']['cqt']['fmin'], 
        n_bins=config['audio']['cqt']['n_bins'], 
        bins_per_octave=config['audio']['cqt']['bins_per_octave']
    ).to(device)

    model = ResNet18Classifier(num_classes=len(train_labels)).to(device)
    
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Resuming training from {resume_checkpoint}...")
        checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    # 4. Training Loop
    epochs = config['training']['epochs']
    patience = config['training'].get('early_stopping_patience', 10)
    
    best_val_acc = 0.0
    epochs_no_improve = 0
    save_path = os.path.join(config['paths']['models_dir'], "resnet18_cnn.pt")
    os.makedirs(config['paths']['models_dir'], exist_ok=True)
    
    for epoch in range(epochs):
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
                aug_config = config.get('augmentation', {})
                if aug_config.get('spec_augment', False):
                    if aug_config.get('minority_only', True):
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
        
        # Save Best Model Checkpoint
        if val_acc > best_val_acc:
            print(f"Validation accuracy improved ({best_val_acc:.2f}% -> {val_acc:.2f}%). Saving checkpoint...")
            best_val_acc = val_acc
            epochs_no_improve = 0
            checkpoint = {
                'state_dict': model.state_dict(),
                'classes': train_labels
            }
            torch.save(checkpoint, save_path)
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation accuracy. Early stopping counter: {epochs_no_improve}/{patience}")
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered! Validation accuracy hasn't improved for {patience} epochs.")
                break
            
    print(f"Training complete! Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Best model saved to {save_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Soundtrack Classifier CNN")
    parser.add_argument("--resume", type=str, default=None, help="Path to a .pt checkpoint to resume training from")
    args = parser.parse_args()
    
    train(resume_checkpoint=args.resume)
