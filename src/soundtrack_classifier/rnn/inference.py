import os
from typing import Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import torch
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Loads configuration from yaml file."""
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            pass
    return {}


def run_streaming_rnn_inference(
    audio_path: str,
    model: torch.nn.Module,
    idx_to_class: Optional[Dict[int, str]] = None,
    sr: Optional[int] = None,
    window_sec: Optional[float] = None,
    hop_sec: Optional[float] = None,
    batch_size: Optional[int] = None,
    device: str = "cpu",
    rms_threshold: Optional[float] = None,
    confidence_threshold: Optional[float] = None,
    min_event_sec: Optional[float] = None,
    cqt_hop: Optional[int] = None,
    include_nonmusic: Optional[bool] = None,
    config: Optional[dict] = None
) -> List[Dict[str, Union[float, str]]]:
    """
    Executes streaming inference across continuous audio using fixed buffers
    with batched GPU inference and Hanning window averaging.
    All parameters are dynamically resolved from config.yaml with override support.
    """
    if config is None:
        config = load_config()

    crnn_cfg = config.get("crnn", {})
    crnn_audio = crnn_cfg.get("audio", {})
    crnn_inf = crnn_cfg.get("inference", {})
    crnn_cqt = crnn_audio.get("cqt", {})

    sr = sr if sr is not None else int(crnn_audio.get("sample_rate", 22050))
    window_sec = window_sec if window_sec is not None else float(crnn_audio.get("window_sec", 10.0))
    hop_sec = hop_sec if hop_sec is not None else float(crnn_audio.get("hop_sec", 4.0))
    batch_size = batch_size if batch_size is not None else int(crnn_inf.get("batch_size", 64))
    rms_threshold = rms_threshold if rms_threshold is not None else float(crnn_inf.get("rms_threshold", 0.005))
    confidence_threshold = confidence_threshold if confidence_threshold is not None else float(crnn_inf.get("confidence_threshold", 0.35))
    min_event_sec = min_event_sec if min_event_sec is not None else float(crnn_inf.get("min_event_sec", 1.0))
    cqt_hop = cqt_hop if cqt_hop is not None else int(crnn_cqt.get("hop_length", 512))
    include_nonmusic = include_nonmusic if include_nonmusic is not None else bool(crnn_inf.get("include_nonmusic", False))

    if idx_to_class is None:
        train_labels = config.get("train_labels", [
            "sheng", "dizi", "erhu", "pipa", "suona", "guzheng",
            "yangqin", "percussion", "strings", "woodwind", "brass", "nonmusic"
        ])
        idx_to_class = {i: c for i, c in enumerate(train_labels)}

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

    slice_starts = list(range(0, total_samples - window_samples + 1, hop_samples))
    print(f"[Streaming RNN Inference] Streaming {len(slice_starts)} windows ({window_sec}s with {hop_sec}s hop, batch_size={batch_size})...")
    
    hann_cached = None

    # Stream through audio in parallel batches
    for b_start in range(0, len(slice_starts), batch_size):
        b_indices = slice_starts[b_start : b_start + batch_size]
        valid_chunks = []
        valid_starts = []

        for start_idx in b_indices:
            chunk = audio[start_idx : start_idx + window_samples]

            # Fast vectorized RMS energy check
            rms_val = float(np.sqrt(np.mean(chunk ** 2)))
            if rms_val < rms_threshold:
                continue

            valid_chunks.append(chunk)
            valid_starts.append(start_idx)

        if not valid_chunks:
            continue

        batch_tensor = torch.tensor(np.array(valid_chunks), dtype=torch.float32, device=device)

        with torch.no_grad():
            frame_logits = model(batch_tensor, return_frames=True)
            batch_probs = torch.softmax(frame_logits, dim=-1).cpu().numpy()

        B, t_chunk, _ = batch_probs.shape
        if hann_cached is None or hann_cached.shape[0] != t_chunk:
            hann_cached = np.hanning(t_chunk)[:, None]

        weighted_probs = batch_probs * hann_cached[None, :, :]

        for i, start_idx in enumerate(valid_starts):
            start_frame = int(round(start_idx / cqt_hop))
            end_frame = min(start_frame + t_chunk, total_frames)
            valid_len = end_frame - start_frame

            accum_probs[start_frame:end_frame] += weighted_probs[i, :valid_len]
            accum_weights[start_frame:end_frame] += hann_cached[:valid_len]

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
