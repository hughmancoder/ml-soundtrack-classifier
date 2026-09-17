from typing import List, Tuple

import librosa
import numpy as np
import torch
from nnAudio.features.cqt import CQT1992v2


def run_sliding_window_inference(
    audio_path: str,
    model: torch.nn.Module,
    idx_to_class: dict,
    sr: int = 44100,
    window_sec: float = 4.0,
    stride_sec: float = 1.0,
    fmin: float = 65.4,
    device: str = 'cpu',
    rms_threshold: float = 0.005
) -> List[Tuple[float, float, str]]:
    """
    Runs sliding window inference across a continuous audio track.
    Includes RMS silence filtering.
    """
    model.eval()

    print(f"Loading {audio_path}...")
    audio, _ = librosa.load(audio_path, sr=sr, mono=True)

    window_samples = int(window_sec * sr)
    stride_samples = int(stride_sec * sr)

    # Pad audio if it's shorter than the minimum window size
    if len(audio) < window_samples:
        padding = window_samples - len(audio)
        audio = np.pad(audio, (0, padding), 'constant')

    cqt = CQT1992v2(sr=sr, fmin=fmin, n_bins=84, bins_per_octave=12).to(device)

    raw_predictions = []
    timestamps_sec = []

    print("Running sliding window inference...")
    for i in range(0, len(audio) - window_samples + 1, stride_samples):
        start_time = i / sr
        chunk = audio[i:i + window_samples]

        # 1. RMS Filter for Silence (skip CNN computation if silent)
        rms_val = np.mean(librosa.feature.rms(y=chunk))
        if rms_val < rms_threshold:
            raw_predictions.append(-1) # -1 represents 'silence' internally
            timestamps_sec.append(start_time)
            continue

        chunk_tensor = torch.tensor(chunk, dtype=torch.float32).to(device)

        with torch.no_grad():
            cqt_spec = cqt(chunk_tensor.unsqueeze(0)).squeeze(0)
            cqt_spec = cqt_spec.unsqueeze(0).unsqueeze(0)

            logits = model(cqt_spec)
            pred_idx = torch.argmax(logits, dim=1).item()

        raw_predictions.append(pred_idx)
        timestamps_sec.append(start_time)

    return _apply_post_processing(raw_predictions, timestamps_sec, idx_to_class, window_sec, stride_sec)

def _categorical_mode_filter(arr: List[int], kernel_size: int = 3) -> np.ndarray:
    """
    Applies a rolling mode (majority vote) filter to categorical data to prevent
    numerical median hallucination on discrete class indices.
    """
    arr_np = np.array(arr)
    result = np.zeros_like(arr_np)
    pad = kernel_size // 2
    padded = np.pad(arr_np, pad, mode='edge')

    for i in range(len(arr_np)):
        window_vals = padded[i:i+kernel_size]
        vals, counts = np.unique(window_vals, return_counts=True)
        result[i] = vals[np.argmax(counts)]

    return result

def _apply_post_processing(
    raw_predictions: List[int],
    timestamps_sec: List[float],
    idx_to_class: dict,
    window_sec: float,
    stride_sec: float
) -> List[Tuple[float, float, str]]:
    """
    Applies Mode Filtering, Contiguous Merging, and Duration Thresholding.
    """
    if not raw_predictions:
        return []

    # 2. Smoothing: Categorical Mode Filtering (replaces median filter)
    smoothed_preds = _categorical_mode_filter(raw_predictions, kernel_size=3)

    # 3. Contiguous Merging
    merged_events = []
    current_label = smoothed_preds[0]
    start_time = timestamps_sec[0]

    for i in range(1, len(smoothed_preds)):
        if smoothed_preds[i] != current_label:
            # Use the start of the next window as the exact boundary to prevent overlapping events
            end_time = timestamps_sec[i]

            # 4. Heuristic Duration Thresholding: Must be >= 1.0s and not be pure silence (-1)
            if (end_time - start_time) >= 1.0 and current_label != -1:
                class_str = idx_to_class.get(current_label, "unknown")
                # Drop nonmusic from the final instrument log if desired. We'll leave it in for context unless instructed otherwise.
                merged_events.append((start_time, end_time, class_str))

            current_label = smoothed_preds[i]
            start_time = timestamps_sec[i]

    # Add final event
    end_time = timestamps_sec[-1] + window_sec
    if (end_time - start_time) >= 1.0 and current_label != -1:
        class_str = idx_to_class.get(current_label, "unknown")
        merged_events.append((start_time, end_time, class_str))

    return merged_events
