import os
import sys
import argparse
import yaml
import torch
import librosa
import numpy as np
from pathlib import Path
from nnAudio.features.cqt import CQT1992v2
from scipy.signal import medfilt

from soundtrack_classifier.cnn.resnet import ResNet18Classifier

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_inference(input_file: str, model_path: str = None):
    # 1. Configuration & Model Loading
    config = load_config()
    
    device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_name)
    print(f"Running inference on device: {device}")

    if model_path is None:
        model_path = os.path.join(config['paths']['models_dir'], "resnet18_cnn.pt")
        
    if not os.path.exists(model_path):
        print(f"Model weights not found at {model_path}. Please train the model first.")
        return

    # Load checkpoint to get both weights and potentially the saved classes
    print(f"Loading weights from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    labels = config['train_labels'] # default from config
    state_dict = checkpoint
    
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        if 'classes' in checkpoint:
            labels = checkpoint['classes']
            print("Loaded class mapping from model metadata.")
        else:
            print("No class metadata found in model. Using config.yaml classes.")
    else:
        print("Legacy model format detected. Using config.yaml classes.")
        
    idx_to_class = {i: cls_name for i, cls_name in enumerate(labels)}

    model = ResNet18Classifier(num_classes=len(labels)).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # 2. CQT Extractor Setup
    sr = config['audio']['sample_rate']
    duration = config['audio']['duration_sec']
    window_samples = int(duration * sr)
    stride_samples = int(1.0 * sr) # 1 second stride
    
    cqt_extractor = CQT1992v2(
        sr=sr, 
        fmin=config['audio']['cqt']['fmin'], 
        n_bins=config['audio']['cqt']['n_bins'], 
        bins_per_octave=config['audio']['cqt']['bins_per_octave']
    ).to(device)

    # 3. Load full continuous audio track
    print(f"Loading full audio track: {input_file}")
    audio, _ = librosa.load(input_file, sr=sr, mono=True)
    num_samples = len(audio)
    
    raw_predictions = []
    timestamps_sec = []

    print("Running sliding window inference...")
    # 4. Sliding Window Extraction
    with torch.no_grad():
        for start_idx in range(0, num_samples - window_samples + 1, stride_samples):
            end_idx = start_idx + window_samples
            chunk = audio[start_idx:end_idx]
            
            audio_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Extract CQT
            cqt_spec = cqt_extractor(audio_tensor)
            cqt_spec = cqt_spec.unsqueeze(1) # [1, 1, Freq, Time]
            
            # Predict
            logits = model(cqt_spec)
            pred_idx = torch.argmax(logits, dim=-1).item()
            
            raw_predictions.append(pred_idx)
            timestamps_sec.append(start_idx / sr)

    if not raw_predictions:
        print("Audio file too short for inference.")
        return

    # 5. Algorithmic Post-Processing
    print("Applying post-processing (median filtering & contiguous merging)...")
    
    # Median Filtering (size 3) to remove single-frame misclassifications
    smoothed_preds = medfilt(raw_predictions, kernel_size=3).astype(int)
    
    # Contiguous Merging
    merged_events = []
    current_label = smoothed_preds[0]
    start_time = timestamps_sec[0]
    
    for i in range(1, len(smoothed_preds)):
        if smoothed_preds[i] != current_label:
            end_time = timestamps_sec[i] + (duration - 1.0) # approx end of window
            
            # Heuristic Duration Thresholding: Must be > 0.5s (which is true given 1s stride, but for structure)
            if (end_time - start_time) >= 0.5:
                merged_events.append({
                    "instrument": idx_to_class[current_label],
                    "start": start_time,
                    "end": end_time
                })
            
            current_label = smoothed_preds[i]
            start_time = timestamps_sec[i]
            
    # Add final event
    end_time = timestamps_sec[-1] + duration
    merged_events.append({
        "instrument": idx_to_class[current_label],
        "start": start_time,
        "end": end_time
    })

    # 6. Output Generation
    print("\n--- Final Extracted Acoustic Events ---")
    for event in merged_events:
        print(f"[{event['start']:05.1f}s - {event['end']:05.1f}s] : {event['instrument']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run sliding window inference using ResNet-18 CNN")
    parser.add_argument("input_file", type=str, help="Path to the legacy film audio file (.wav)")
    parser.add_argument("--model", type=str, default=None, help="Path to trained CNN model weights")
    args = parser.parse_args()
    
    run_inference(args.input_file, args.model)
