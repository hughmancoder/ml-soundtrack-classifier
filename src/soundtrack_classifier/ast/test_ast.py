"""
Test AST Model Against Film Cue Sheet Ground Truth Annotations.

Runs the ContinuousFilmScorer across film audio, maps predictions against
timestamped cue sheet ground truth, calculates polyphonic dominant-hit
accuracy (Top-1 and Top-3), non-music rejection rates, and per-class precision/recall,
and saves detailed evaluation results to a JSON log file.
"""

import argparse
import json
import os
import time
from datetime import datetime
from typing import Optional

import numpy as np
import soundfile as sf
import torch

from soundtrack_classifier.ast.pipeline import load_config
from soundtrack_classifier.ast.scoring import ContinuousFilmScorer
from soundtrack_classifier.evaluation_utils import (
    align_and_compute_metrics,
    load_cue_sheet,
    prepare_audio_file,
)


def run_ast_evaluation(
    audio_path: Optional[str] = None,
    model_path: Optional[str] = None,
    cuesheet_path: Optional[str] = None,
    mapping_path: Optional[str] = None,
    output_path: Optional[str] = None,
    window_sec: Optional[float] = None,
    stride_sec: Optional[float] = None,
    rms_threshold: Optional[float] = None,
    batch_size: Optional[int] = None,
    device_str: Optional[str] = None,
    config_path: str = "config/config.yaml",
):
    start_run_time = time.time()
    cfg = load_config(config_path)
    ast_cfg = cfg.get("ast", {})
    paths_cfg = ast_cfg.get("paths", {})
    scoring_cfg = ast_cfg.get("scoring", {})
    feature_cfg = ast_cfg.get("feature_extraction", {})

    audio_path = (
        audio_path
        or paths_cfg.get("film_audio_path")
        or ast_cfg.get("separation", {}).get("film_audio_path")
        or "data/test/come-drink-with-me/Come Drink With Me.wav"
    )
    if not os.path.exists(audio_path) and os.path.exists("data/raw/Come Drink With Me.mp4"):
        audio_path = "data/raw/Come Drink With Me.mp4"

    model_path = model_path or paths_cfg.get("model_checkpoint", "models/ast_probing_head.pt")
    cuesheet_path = (
        cuesheet_path
        or paths_cfg.get("cuesheet")
        or "data/test/come-drink-with-me/Film Music Cue Sheet for Come Drink with Me (English) - English Version.csv"
    )
    mapping_path = mapping_path or paths_cfg.get("mapping", "data/test/come-drink-with-me/instrument_mapping.json")
    output_path = output_path or paths_cfg.get("annotation_json", "reports/ast_inference_eval_come_drink_with_me.json")

    window_sec = float(window_sec if window_sec is not None else scoring_cfg.get("window_sec", 4.0))
    stride_sec = float(stride_sec if stride_sec is not None else scoring_cfg.get("hop_sec", 1.0))
    batch_size = int(batch_size if batch_size is not None else scoring_cfg.get("batch_size", 32))
    rms_threshold = float(rms_threshold if rms_threshold is not None else scoring_cfg.get("rms_threshold", 0.008))
    device_str = device_str or feature_cfg.get("device", "mps")

    device = torch.device(device_str if torch.backends.mps.is_available() or torch.cuda.is_available() else "cpu")
    print(f"[*] Running AST Cue Sheet Evaluation on device: {device}")
    print(f"[*] Evaluation parameters from config ({config_path}):")
    print(f"    - window={window_sec}s, stride={stride_sec}s, rms_threshold={rms_threshold}, batch_size={batch_size}")
    print(f"    - audio={audio_path}")
    print(f"    - model={model_path}")
    print(f"    - output={output_path}")

    # 1. Load instrument mapping
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Instrument mapping JSON not found: {mapping_path}")
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)
    inst_mapping = mapping_data.get("instrument_mapping", mapping_data)

    # 2. Load cue sheet
    print(f"[*] Loading cue sheet: {cuesheet_path}")
    cues = load_cue_sheet(cuesheet_path, inst_mapping)
    valid_cues = [c for c in cues if c["ground_truth_classes"]]
    print(f"    Loaded {len(cues)} annotated cues ({len(valid_cues)} with valid ground truth classes).")

    # 3. Prepare audio file
    wav_file = prepare_audio_file(audio_path)

    # Load audio to calculate duration and RMS energy later
    print(f"[*] Reading audio from {wav_file} to calculate RMS masking...")
    audio, file_sr = sf.read(wav_file)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    total_audio_sec = len(audio) / file_sr

    # 4. Score film with AST ContinuousFilmScorer
    scorer = ContinuousFilmScorer(
        probing_head_checkpoint=model_path,
        window_sec=window_sec,
        hop_sec=stride_sec,
        batch_size=batch_size,
        device=device_str
    )

    print("[*] Running Phase 6 AST Feature Extraction & Probing...")
    probs_matrix, timestamps = scorer.score_audio_file(wav_file, output_probabilities_path=None)

    # 5. Load classes
    labels = cfg.get("train_labels", [
        "erhu", "pipa", "dizi", "percussion", "strings_western", "sheng",
        "suona", "guzheng", "yangqin", "woodwind", "brass", "nonmusic"
    ])

    try:
        chkpt = torch.load(model_path, map_location="cpu", weights_only=False)
        if "classes" in chkpt:
            labels = chkpt["classes"]
    except Exception:
        pass

    idx_to_class = {i: c for i, c in enumerate(labels)}

    # 6. Format window results and apply RMS thresholding
    window_results = []
    print("[*] Processing probabilities and applying RMS thresholding...")

    for i in range(len(timestamps)):
        t_start = timestamps[i]
        t_mid = t_start + (window_sec / 2.0)
        t_end = t_start + window_sec

        # Calculate RMS for this window to mimic CNN/CRNN logic
        s_idx = int(t_start * file_sr)
        e_idx = int(t_end * file_sr)
        chunk = audio[s_idx:e_idx]
        if len(chunk) == 0:
            rms_energy = 0.0
        else:
            rms_energy = float(np.sqrt(np.mean(chunk ** 2)))

        is_silent = rms_energy < rms_threshold

        p = probs_matrix[i]
        sorted_indices = np.argsort(p)[::-1]
        top1_idx = int(sorted_indices[0])
        top3_indices = [int(x) for x in sorted_indices[:3]]

        if is_silent:
            pred_class = "nonmusic"
            pred_prob = 1.0
            top3_classes = ["nonmusic", idx_to_class[top1_idx], idx_to_class[top3_indices[1]]]
        else:
            pred_class = idx_to_class[top1_idx]
            pred_prob = float(p[top1_idx])
            top3_classes = [idx_to_class[idx] for idx in top3_indices]

        window_results.append({
            "window_index": i,
            "start_sec": round(t_start, 2),
            "end_sec": round(t_end, 2),
            "mid_sec": round(t_mid, 2),
            "rms_energy": round(rms_energy, 6),
            "is_silent": is_silent,
            "predicted_class": pred_class,
            "confidence": round(pred_prob, 4),
            "top3_classes": top3_classes
        })

    # 7. Alignment and metrics computation
    print("[*] Aligning predictions with cue sheet ground truth...")
    overall_metrics, per_cue_summary, per_class_summary, window_results = align_and_compute_metrics(
        window_results, cues, labels
    )

    # 8. Report Generation
    report = {
        "metadata": {
            "model_path": model_path,
            "audio_path": audio_path,
            "wav_processed": wav_file,
            "cuesheet_path": cuesheet_path,
            "mapping_path": mapping_path,
            "evaluation_timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(time.time() - start_run_time, 2),
            "window_sec": window_sec,
            "stride_sec": stride_sec,
            "rms_threshold": rms_threshold,
            "device": str(device),
            "total_audio_duration_sec": round(total_audio_sec, 2),
            "total_windows": len(window_results)
        },
        "overall_metrics": {
            "music_cues": overall_metrics["music_cues"],
            "non_music": overall_metrics["nonmusic_cues"],
            "total_film": {
                "total_windows": overall_metrics["total_windows_analyzed"],
                "overall_accuracy_percent": overall_metrics["total_accuracy_percent"]
            }
        },
        "per_class_metrics": per_class_summary,
        "per_cue_evaluations": per_cue_summary,
        "window_predictions_sample": window_results[:50]
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"\n[+] Saved AST evaluation report to: {output_path}")
    print(f"    - Total Windows: {len(window_results)}")
    print(f"    - Total Accuracy: {overall_metrics['total_accuracy_percent']}%")
    print(f"    - Top-1 Music Hit Rate: {overall_metrics['music_cues']['top1_dominant_accuracy_percent']}%")
    print(f"    - Top-3 Music Hit Rate: {overall_metrics['music_cues']['top3_dominant_accuracy_percent']}%")


def main():
    cfg = load_config("config/config.yaml")
    ast_cfg = cfg.get("ast", {})
    paths_cfg = ast_cfg.get("paths", {})
    scoring_cfg = ast_cfg.get("scoring", {})
    feature_cfg = ast_cfg.get("feature_extraction", {})

    default_audio = (
        paths_cfg.get("film_audio_path")
        or ast_cfg.get("separation", {}).get("film_audio_path")
        or "data/test/come-drink-with-me/Come Drink With Me.wav"
    )
    if not os.path.exists(default_audio) and os.path.exists("data/raw/Come Drink With Me.mp4"):
        default_audio = "data/raw/Come Drink With Me.mp4"

    default_model = paths_cfg.get("model_checkpoint", "models/ast_probing_head.pt")
    default_cuesheet = (
        paths_cfg.get("cuesheet")
        or "data/test/come-drink-with-me/Film Music Cue Sheet for Come Drink with Me (English) - English Version.csv"
    )
    default_mapping = paths_cfg.get("mapping", "data/test/come-drink-with-me/instrument_mapping.json")
    default_output = paths_cfg.get("annotation_json", "reports/ast_inference_eval_come_drink_with_me.json")

    default_window = float(scoring_cfg.get("window_sec", 4.0))
    default_stride = float(scoring_cfg.get("hop_sec", 1.0))
    default_batch = int(scoring_cfg.get("batch_size", 32))
    default_rms = float(scoring_cfg.get("rms_threshold", 0.008))
    default_device = feature_cfg.get("device", "mps")

    parser = argparse.ArgumentParser(description="Evaluate AST Pipeline against Cue Sheet")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--audio", type=str, default=default_audio, help=f"Path to audio/video file (default: {default_audio})")
    parser.add_argument("--model", type=str, default=default_model, help=f"Model checkpoint path (default: {default_model})")
    parser.add_argument("--cuesheet", type=str, default=default_cuesheet, help="Path to cue sheet CSV")
    parser.add_argument("--mapping", type=str, default=default_mapping, help="Path to instrument mapping JSON")
    parser.add_argument("--output", type=str, default=default_output, help="Path to output JSON report")
    parser.add_argument("--window", type=float, default=default_window, help=f"Window duration in seconds (default: {default_window})")
    parser.add_argument("--stride", type=float, default=default_stride, help=f"Stride duration in seconds (default: {default_stride})")
    parser.add_argument("--rms", type=float, default=default_rms, help=f"RMS silence cutoff threshold (default: {default_rms})")
    parser.add_argument("--batch", type=int, default=default_batch, help=f"Inference batch size (default: {default_batch})")
    parser.add_argument("--device", type=str, default=default_device, help=f"Device for inference (default: {default_device})")

    args = parser.parse_args()

    run_ast_evaluation(
        audio_path=args.audio,
        model_path=args.model,
        cuesheet_path=args.cuesheet,
        mapping_path=args.mapping,
        output_path=args.output,
        window_sec=args.window,
        stride_sec=args.stride,
        rms_threshold=args.rms,
        batch_size=args.batch,
        device_str=args.device,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
