import torch
import librosa
import numpy as np
from typing import List, Tuple
from nnAudio.features.cqt import CQT1992v2

def run_sliding_window_inference(
    audio_path: str, 
    model: torch.nn.Module, 
    idx_to_class: dict,
    sr: int = 44100, 
    window_sec: float = 3.0, 
    stride_sec: float = 1.0, 
    device: str = 'cpu'
) -> List[Tuple[float, float, str]]:
    """
    Runs sliding window inference across a continuous audio track.
    Args:
        audio_path: Path to the full film audio.
        model: The trained ResNet-18 CNN classifier.
        idx_to_class: Mapping from integer class index to instrument string.
        sr: Sample rate.
        window_sec: Size of the sliding window in seconds (must match training).
        stride_sec: How far to slide the window each step.
    Returns:
        List of continuous timestamps in format [(start_time, end_time, "Instrument"), ...]
    """
    model.eval()
    
    print(f"Loading {audio_path}...")
    audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    
    window_samples = int(window_sec * sr)
    stride_samples = int(stride_sec * sr)
    
    # Initialize CQT extractor
    cqt = CQT1992v2(sr=sr, fmin=32.7, n_bins=84, bins_per_octave=12).to(device)
    
    predictions = []
    
    print("Running sliding window inference...")
    # Slide across the audio
    for i in range(0, len(audio) - window_samples + 1, stride_samples):
        start_time = i / sr
        chunk = audio[i:i + window_samples]
        
        # Prepare tensor
        chunk_tensor = torch.tensor(chunk, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            # Compute CQT dynamically: shape [1, F, T]
            cqt_spec = cqt(chunk_tensor.unsqueeze(0)).squeeze(0)
            # Add batch and channel dims: shape [1, 1, F, T]
            cqt_spec = cqt_spec.unsqueeze(0).unsqueeze(0)
            
            # Predict
            logits = model(cqt_spec)
            pred_idx = torch.argmax(logits, dim=1).item()
            pred_class = idx_to_class[pred_idx]
            
        predictions.append((start_time, pred_class))
        
    return _merge_predictions(predictions, window_sec, stride_sec)

def _merge_predictions(predictions: List[Tuple[float, str]], window_sec: float, stride_sec: float) -> List[Tuple[float, float, str]]:
    """
    Merges consecutive identical predictions into start/end timestamps.
    """
    if not predictions:
        return []
        
    timestamps = []
    current_class = predictions[0][1]
    start_time = predictions[0][0]
    
    for i in range(1, len(predictions)):
        time, pred_class = predictions[i]
        
        if pred_class != current_class:
            # The class changed. End the previous timestamp.
            # End time is the start of the last matching window + window duration.
            end_time = predictions[i-1][0] + window_sec
            timestamps.append((start_time, end_time, current_class))
            
            # Start new tracking
            current_class = pred_class
            start_time = time
            
    # Add the final tracked sequence
    final_time = predictions[-1][0] + window_sec
    timestamps.append((start_time, final_time, current_class))
    
    return timestamps
