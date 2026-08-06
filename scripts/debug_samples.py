import os
import glob
import yaml
import torch
import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from nnAudio.features.cqt import CQT1992v2
import random

from soundtrack_classifier.ingestion.degradation import SyntheticDegradation

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def debug_samples():
    config = load_config()
    data_dir = config['paths']['data_dir']
    sample_dir = "data/sample"
    os.makedirs(sample_dir, exist_ok=True)
    
    sr = config['audio']['sample_rate']
    
    # 1. Setup extractors and degraders
    degrader = SyntheticDegradation(sr=sr)
    
    device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_name)
    
    cqt_extractor = CQT1992v2(
        sr=sr, 
        fmin=config['audio']['cqt']['fmin'], 
        n_bins=config['audio']['cqt']['n_bins'], 
        bins_per_octave=config['audio']['cqt']['bins_per_octave']
    ).to(device)

    # 2. Pick a couple of random files
    all_files = glob.glob(os.path.join(data_dir, "*", "*.wav"))
    if not all_files:
        print(f"No .wav files found in {data_dir}. Run slicing script first.")
        return
        
    sample_files = random.sample(all_files, min(3, len(all_files)))
    
    for i, file_path in enumerate(sample_files):
        print(f"\nProcessing Debug Sample {i+1}: {file_path}")
        instrument = Path(file_path).parent.name
        base_name = Path(file_path).stem
        
        # Load clean audio
        clean_audio, _ = librosa.load(file_path, sr=sr, mono=True)
        
        # Ensure it's 3 seconds long for consistent CQT
        target_len = int(3.0 * sr)
        if len(clean_audio) < target_len:
            clean_audio = np.pad(clean_audio, (0, target_len - len(clean_audio)))
        else:
            clean_audio = clean_audio[:target_len]
        
        # Generate degraded audio
        print(f"  Applying pedalboard degradation...")
        degraded_audio = degrader.apply(clean_audio)
        
        # Save WAVs
        clean_out_path = os.path.join(sample_dir, f"{instrument}_{base_name}_clean.wav")
        degraded_out_path = os.path.join(sample_dir, f"{instrument}_{base_name}_degraded.wav")
        sf.write(clean_out_path, clean_audio, sr)
        sf.write(degraded_out_path, degraded_audio, sr)
        
        # Generate CQTs
        print(f"  Computing CQTs...")
        with torch.no_grad():
            clean_tensor = torch.tensor(clean_audio, dtype=torch.float32).unsqueeze(0).to(device)
            degraded_tensor = torch.tensor(degraded_audio, dtype=torch.float32).unsqueeze(0).to(device)
            
            clean_cqt = cqt_extractor(clean_tensor).squeeze().cpu().numpy()
            degraded_cqt = cqt_extractor(degraded_tensor).squeeze().cpu().numpy()
            
        # Plot CQTs side by side
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.title(f"Clean CQT ({instrument})")
        plt.imshow(clean_cqt, aspect='auto', origin='lower', cmap='viridis')
        plt.xlabel('Time')
        plt.ylabel('CQT Bin')
        plt.colorbar(format='%+2.0f dB')
        
        plt.subplot(1, 2, 2)
        plt.title(f"Degraded CQT ({instrument})")
        plt.imshow(degraded_cqt, aspect='auto', origin='lower', cmap='viridis')
        plt.xlabel('Time')
        plt.ylabel('CQT Bin')
        plt.colorbar(format='%+2.0f dB')
        
        plt.tight_layout()
        img_out_path = os.path.join(sample_dir, f"{instrument}_{base_name}_comparison.png")
        plt.savefig(img_out_path)
        plt.close()
        
        print(f"  Saved comparison image to: {img_out_path}")

if __name__ == "__main__":
    debug_samples()
