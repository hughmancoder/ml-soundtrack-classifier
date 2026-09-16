import os
from typing import Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import torch


def run_streaming_rnn_inference(
    audio_path: str,
    model: torch.nn.Module,
    idx_to_class: Dict[int, str],
    sr: int = 22050,
    window_sec: float = 10.0,
    hop_sec: float = 2.5,
    device: str = "cpu",
    rms_threshold: float = 0.005,
    confidence_threshold: float = 0.35,
    min_event_sec: float = 1.0,
    cqt_hop: int = 512,
    include_nonmusic: bool = False
) -> List[Dict[str, Union[float, str]]]:
    """
    Executes streaming inference across continuous audio using 10-second fixed buffers
    with 50% to 75% overlap and Hanning window averaging.

    Operational Design:
    - 10-Second Fixed Buffer:
        Provides the recurrent GRU units with adequate temporal context to model extended
        phrasing (erhu portamentos, guqin decays) and distinguish steady-state optical film noise.
    - 50%-75% Overlap (hop_sec = 2.5s -> 75% overlap):
        Dense evaluation across continuous boundaries.
    - Hanning Window Averaging:
        Applies a smooth Hanning tapering window across each 10s prediction segment, eliminating
        boundary padding artifacts and transition edge discontinuities.
    """
    model.eval()
    model.to(device)

    print(f"[Streaming RNN Inference] Loading {audio_path} at {sr} Hz...")
    audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    total_samples = len(audio)
    total_duration = total_samples / sr
    print(f"[Streaming RNN Inference] Total duration: {total_duration:.2f}s ({total_samples} samples)")

    window_samples = int(window_sec * sr)
    hop_samples = int(hop_sec * sr)

    # Pad if shorter than 1 window
    if total_samples < window_samples:
        padding = window_samples - total_samples
        audio = np.pad(audio, (0, padding), mode="constant")
        total_samples = len(audio)

    total_frames = int(np.ceil(total_samples / cqt_hop))
    num_classes = len(idx_to_class)

    accum_probs = np.zeros((total_frames, num_classes), dtype=np.float32)
    accum_weights = np.zeros((total_frames, 1), dtype=np.float32)

    # Stream through audio buffer with overlap
    print(f"[Streaming RNN Inference] Streaming {window_sec}s windows with {hop_sec}s hop (overlap: {(1 - hop_sec/window_sec)*100:.0f}%)...")
    
    for start_idx in range(0, total_samples - window_samples + 1, hop_samples):
        chunk = audio[start_idx : start_idx + window_samples]

        # Check silence via RMS energy
        rms_val = np.mean(librosa.feature.rms(y=chunk))
        if rms_val < rms_threshold:
            continue

        chunk_tensor = torch.tensor(chunk, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            # Get frame-level predictions (1, T_chunk, num_classes)
            frame_logits = model(chunk_tensor, return_frames=True)
            frame_probs = torch.softmax(frame_logits, dim=-1).squeeze(0).cpu().numpy()

        t_chunk = frame_probs.shape[0]
        # Hanning window weighting to suppress edge/padding effects
        hann = np.hanning(t_chunk)[:, None]

        start_frame = int(round(start_idx / cqt_hop))
        end_frame = min(start_frame + t_chunk, total_frames)
        valid_len = end_frame - start_frame

        accum_probs[start_frame:end_frame] += (frame_probs[:valid_len] * hann[:valid_len])
        accum_weights[start_frame:end_frame] += hann[:valid_len]

    # Normalize by accumulated weights
    valid_mask = (accum_weights > 1e-6).squeeze()
    if np.any(valid_mask):
        accum_probs[valid_mask] /= accum_weights[valid_mask]

    # Frame-to-event decoding
    frame_dur = cqt_hop / sr
    events = []

    # Frame-level argmax & confidence thresholding
    pred_classes = np.argmax(accum_probs, axis=1)
    confidences = np.max(accum_probs, axis=1)

    # Invalidate frames where weights were 0 or confidence < threshold
    invalid_mask = (~valid_mask) | (confidences < confidence_threshold)
    pred_classes[invalid_mask] = -1

    # Merge contiguous segments
    if len(pred_classes) > 0:
        curr_label = pred_classes[0]
        curr_start = 0.0
        curr_conf_sum = confidences[0]
        curr_count = 1

        for i in range(1, len(pred_classes)):
            lbl = pred_classes[i]
            if lbl == curr_label:
                curr_conf_sum += confidences[i]
                curr_count += 1
            else:
                curr_end = i * frame_dur
                event_duration = curr_end - curr_start
                if curr_label != -1 and event_duration >= min_event_sec:
                    class_name = idx_to_class.get(curr_label, "unknown")
                    if include_nonmusic or class_name != "nonmusic":
                        events.append({
                            "start_sec": round(float(curr_start), 2),
                            "end_sec": round(float(curr_end), 2),
                            "duration_sec": round(float(event_duration), 2),
                            "instrument": class_name,
                            "confidence": round(float(curr_conf_sum / max(1, curr_count)), 3)
                        })
                curr_label = lbl
                curr_start = i * frame_dur
                curr_conf_sum = confidences[i]
                curr_count = 1

        # Final segment
        curr_end = len(pred_classes) * frame_dur
        event_duration = curr_end - curr_start
        if curr_label != -1 and event_duration >= min_event_sec:
            class_name = idx_to_class.get(curr_label, "unknown")
            if include_nonmusic or class_name != "nonmusic":
                events.append({
                    "start_sec": round(float(curr_start), 2),
                    "end_sec": round(float(curr_end), 2),
                    "duration_sec": round(float(event_duration), 2),
                    "instrument": class_name,
                    "confidence": round(float(curr_conf_sum / max(1, curr_count)), 3)
                })

    print(f"[Streaming RNN Inference] Detected {len(events)} musical events.")
    return events
