import os
from typing import Any, Callable, Dict, List, Optional

import torch

from soundtrack_classifier.ast.pipeline import load_config
from soundtrack_classifier.ast.postprocessing import TemporalPostProcessor
from soundtrack_classifier.ast.probing_head import ProbingHead
from soundtrack_classifier.ast.scoring import ContinuousFilmScorer
from soundtrack_classifier.evaluation_utils import prepare_audio_file


def run_ast_inference(
    audio_path: str,
    model_path: Optional[str] = None,
    class_labels: Optional[List[str]] = None,
    window_sec: Optional[float] = None,
    hop_sec: Optional[float] = None,
    batch_size: Optional[int] = None,
    device: Optional[str] = None,
    p_high: Optional[float] = None,
    p_low: Optional[float] = None,
    median_filter_sec: Optional[float] = None,
    min_duration_sec: Optional[float] = None,
    include_nonmusic: bool = False,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    config: Optional[dict] = None,
) -> List[Dict[str, Any]]:
    """
    Executes end-to-end Audio Spectrogram Transformer (AST) / Foundation Model Probing inference.

    Streams continuous audio, extracts dual-resolution embeddings via MERT (24kHz acoustic-music)
    and CLAP (48kHz contrastive audio-text), classifies using the multi-layer probing head,
    and applies hysteresis thresholding and temporal median smoothing to emit clean acoustic events.

    Args:
        audio_path: Path to input audio or video file.
        model_path: Path to trained probing head checkpoint.
        class_labels: Target class names. Defaults to config train_labels.
        window_sec: Slicing window duration in seconds (default: 4.0s).
        hop_sec: Stride duration between windows in seconds (default: 0.5s).
        batch_size: GPU batch size for foundation model feature extraction (default: 32).
        device: Execution device ('mps', 'cuda', 'cpu').
        p_high: Activation probability threshold for hysteresis trigger (default: 0.65).
        p_low: Deactivation probability threshold for hysteresis release (default: 0.35).
        median_filter_sec: Median filter window in seconds (default: 1.5s).
        min_duration_sec: Minimum event duration threshold (default: 1.0s).
        include_nonmusic: Whether to retain 'nonmusic'/'noise'/'silence' labels in output.
        progress_callback: Optional callback receiving (progress_ratio, status_text).
        config: Optional preloaded configuration dictionary.

    Returns:
        List of event dicts with keys: 'start_time', 'end_time', 'duration', 'instrument', 'confidence'.
    """
    if config is None:
        config = load_config()

    ast_cfg = config.get("ast", {})
    scoring_cfg = ast_cfg.get("scoring", {})
    post_cfg = ast_cfg.get("postprocessing", {})

    if class_labels is None:
        class_labels = config.get("train_labels", [
            "erhu", "pipa", "dizi", "percussion", "strings_western", "sheng",
            "suona", "guzheng", "yangqin", "woodwind", "brass", "nonmusic",
        ])

    window_sec = float(window_sec if window_sec is not None else scoring_cfg.get("window_sec", 4.0))
    hop_sec = float(hop_sec if hop_sec is not None else scoring_cfg.get("hop_sec", 0.5))
    batch_size = int(batch_size if batch_size is not None else scoring_cfg.get("batch_size", 32))
    p_high = float(p_high if p_high is not None else post_cfg.get("p_high", 0.65))
    p_low = float(p_low if p_low is not None else post_cfg.get("p_low", 0.35))
    median_filter_sec = float(median_filter_sec if median_filter_sec is not None else post_cfg.get("median_filter_sec", 1.5))
    min_duration_sec = float(min_duration_sec if min_duration_sec is not None else post_cfg.get("min_duration_sec", 1.0))

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = model_path or "models/ast_probing_head.pt"

    # Prepare audio (extracts audio track if input is video)
    wav_path = prepare_audio_file(audio_path)

    if progress_callback:
        progress_callback(0.15, "Phase 6: Initializing MERT+CLAP foundation models...")

    scorer = ContinuousFilmScorer(
        probing_head_checkpoint=checkpoint_path if os.path.exists(checkpoint_path) else "models/ast_probing_head.pt",
        window_sec=window_sec,
        hop_sec=hop_sec,
        batch_size=batch_size,
        device=device,
    )

    if not os.path.exists(checkpoint_path):
        print(f"Notice: Checkpoint {checkpoint_path} not found. Running with initialized ProbingHead architecture.")
        scorer.probing_head = ProbingHead(num_classes=len(class_labels)).to(device)
        scorer.probing_head.eval()

    if progress_callback:
        progress_callback(0.35, "Phase 6: Extracting fused embeddings and scoring windows...")

    probabilities, timestamps = scorer.score_audio_file(wav_path, output_probabilities_path=None)

    if progress_callback:
        progress_callback(0.80, "Phase 7: Applying temporal smoothing and hysteresis post-processing...")

    processor = TemporalPostProcessor(
        class_labels=class_labels,
        p_high=p_high,
        p_low=p_low,
        median_filter_sec=median_filter_sec,
        min_duration_sec=min_duration_sec,
        hop_sec=hop_sec,
    )
    raw_events = processor.process(probabilities, timestamps)

    formatted_events: List[Dict[str, Any]] = []
    for ev in raw_events:
        inst_name = str(ev["instrument"])
        if not include_nonmusic and inst_name.lower() in ["nonmusic", "noise", "silence"]:
            continue

        start_t = float(ev["start_time"])
        end_t = float(ev["end_time"])
        dur = round(end_t - start_t, 2)
        conf = float(ev["confidence"])

        formatted_events.append({
            "start_time": start_t,
            "end_time": end_t,
            "duration": dur,
            "instrument": inst_name,
            "confidence": conf,
        })

    if progress_callback:
        progress_callback(1.0, "AST inference complete!")

    return formatted_events
