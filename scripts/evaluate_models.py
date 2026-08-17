import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from nnAudio.features.cqt import CQT1992v2
from soundtrack_classifier.cnn.dataset import SoundtrackCNNDataset
from soundtrack_classifier.cnn.resnet import ResNet18Classifier

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def evaluate_models(model_paths):
    config = load_config()
    train_labels = config['train_labels']
    class_to_idx = {cls_name: i for i, cls_name in enumerate(train_labels)}
    
    device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_name)
    print(f"Evaluating on device: {device}\n")

    # Dataset Setup
    data_dir = config['paths']['data_dir']
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} not found.")
        return

    dataset = SoundtrackCNNDataset(
        data_dir=data_dir, 
        class_to_idx=class_to_idx,
        sr=config['audio']['sample_rate'],
        duration_sec=config['audio']['duration_sec'],
        degradation_config=config.get('degradation', None)
    )
    
    # Use EXACT same split seed as training (42) to get the true validation set
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )

    cqt_extractor = CQT1992v2(
        sr=config['audio']['sample_rate'], 
        fmin=config['audio']['cqt']['fmin'], 
        n_bins=config['audio']['cqt']['n_bins'], 
        bins_per_octave=config['audio']['cqt']['bins_per_octave']
    ).to(device)

    model = ResNet18Classifier(num_classes=len(train_labels)).to(device)
    
    print("-" * 60)
    print(f"{'Model File':<30} | {'Validation Accuracy':<20}")
    print("-" * 60)

    for path in model_paths:
        if not os.path.exists(path):
            print(f"{os.path.basename(path):<30} | NOT FOUND")
            continue

        try:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            model.eval()
            
            correct = 0
            total = 0
            
            with torch.no_grad():
                for audio_batch, labels in val_loader:
                    audio_batch = audio_batch.to(device)
                    labels = labels.to(device)
                    
                    cqt_batch = cqt_extractor(audio_batch)
                    cqt_batch = cqt_batch.unsqueeze(1)
                    
                    outputs = model(cqt_batch)
                    _, predicted = torch.max(outputs, 1)
                    
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
            val_acc = 100.0 * correct / total
            print(f"{os.path.basename(path):<30} | {val_acc:.2f}%")
        except Exception as e:
            print(f"{os.path.basename(path):<30} | ERROR: {str(e)}")
            
    print("-" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate multiple CNN checkpoints against the validation set.")
    parser.add_argument("models", nargs="*", default=["models/resnet18_cnn_1.pt", "models/resnet18_cnn_2.pt", "models/resnet18_cnn.pt"], help="Paths to .pt models to evaluate")
    args = parser.parse_args()
    
    evaluate_models(args.models)
