"""
Test CNN Model Against Film Cue Sheet Ground Truth Annotations.

Runs sliding-window inference across film audio, maps predictions against
timestamped cue sheet ground truth, calculates polyphonic dominant-hit accuracy,
and saves detailed evaluation results to a JSON log file.
"""

import argparse
import json
import os
import time
from datetime import datetime
from typing import Any, Dict

import numpy as np
import soundfile as sf
import torch
from nnAudio.features.cqt import CQT1992v2

from soundtrack_classifier.cnn.resnet import ResNet50Classifier
from soundtrack_classifier.evaluation_utils import (
    align_and_compute_metrics,
    load_cue_sheet,
    prepare_audio_file,
)


def run_evaluation(
    audio_path: str,
    model_path: str,
    cuesheet_path: str,
    mapping_path: str,
    output_path: str,
    window_sec: float = 4.0,
    stride_sec: float = 1.0,
    batch_size: int = 64,
    fmin: float = 65.4,
    rms_threshold: float = 0.001,
    device_str: str = None,
    save_all_windows: bool = False
) -> Dict[str, Any]:
    """
    Executes batched sliding-window inference and evaluates against cue sheet annotations.
    """
    start_run_time = time.time()

    # 1. Device selection
    if device_str is None:
        device_str = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"[*] Running CNN Cue Sheet Evaluation on device: {device}")

    # 2. Load instrument mapping
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Instrument mapping JSON not found: {mapping_path}")
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)

    inst_mapping = mapping_data.get("instrument_mapping", {})

    # 3. Load cue sheet
    print(f"[*] Loading cue sheet: {cuesheet_path}")
    cues = load_cue_sheet(cuesheet_path, inst_mapping)
    print(f"    Loaded {len(cues)} annotated cues.")

    # 4. Prepare and load audio
    wav_file = prepare_audio_file(audio_path, target_sr=44100)
    print(f"[*] Reading audio from: {wav_file}...")
    audio, sr = sf.read(wav_file)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)  # Ensure mono
    total_audio_samples = len(audio)
    total_audio_sec = total_audio_samples / sr
    print(f"    Loaded {total_audio_sec/60:.2f} minutes ({total_audio_sec:.2f}s) of audio at {sr} Hz.")

    # 5. Load model checkpoint
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    print(f"[*] Loading model checkpoint from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    labels = [
        "erhu", "pipa", "dizi", "percussion", "strings_western", "sheng",
        "suona", "guzheng", "yangqin", "woodwind", "brass", "nonmusic"
    ]
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict", checkpoint)
        if "classes" in checkpoint:
            labels = checkpoint["classes"]
    else:
        state_dict = checkpoint

    try:
        model = ResNet50Classifier(num_classes=len(labels)).to(device)
        model.load_state_dict(state_dict)
    except Exception:
        print("[!] Checkpoint does not match ResNet-50. Attempting ResNet-18 legacy architecture...")
        from soundtrack_classifier.cnn.resnet import ResNet18Classifier
        model = ResNet18Classifier(num_classes=len(labels)).to(device)
        model.load_state_dict(state_dict)

    model.eval()

    idx_to_class = {i: c for i, c in enumerate(labels)}

    # 6. Setup CQT feature extractor
    cqt = CQT1992v2(sr=sr, fmin=fmin, n_bins=84, bins_per_octave=12).to(device)

    # 7. Generate sliding windows
    window_samples = int(window_sec * sr)
    stride_samples = int(stride_sec * sr)
    total_windows = (total_audio_samples - window_samples) // stride_samples + 1
    print(f"[*] Sliced into {total_windows} sliding windows (window={window_sec}s, stride={stride_sec}s).")

    # 8. Batched Inference
    print(f"[*] Running batched inference (batch_size={batch_size})...")
    window_results = []

    batch_chunks = []
    batch_meta = []

    for w_idx in range(total_windows):
        start_sample = w_idx * stride_samples
        end_sample = start_sample + window_samples
        chunk = audio[start_sample:end_sample]

        start_time = start_sample / sr
        end_time = end_sample / sr
        mid_time = start_time + (window_sec / 2.0)

        # Check RMS silence
        rms_energy = float(np.sqrt(np.mean(chunk ** 2)))
        is_silent = rms_energy < rms_threshold

        batch_chunks.append(chunk)
        batch_meta.append({
            "window_index": w_idx,
            "start_sec": round(start_time, 2),
            "end_sec": round(end_time, 2),
            "mid_sec": round(mid_time, 2),
            "rms_energy": round(rms_energy, 6),
            "is_silent": is_silent
        })

        if len(batch_chunks) == batch_size or w_idx == total_windows - 1:
            batch_tensor = torch.tensor(np.array(batch_chunks), dtype=torch.float32, device=device)

            with torch.no_grad():
                spec = cqt(batch_tensor).unsqueeze(1)
                logits = model(spec)
                probs = torch.softmax(logits, dim=1).cpu().numpy()

            for i, meta in enumerate(batch_meta):
                p = probs[i]
                sorted_indices = np.argsort(p)[::-1]
                top1_idx = int(sorted_indices[0])
                top3_indices = [int(x) for x in sorted_indices[:3]]

                # If window is pure silence, dominant label is nonmusic
                if meta["is_silent"]:
                    pred_class = "nonmusic"
                    pred_prob = 1.0
                    top3_classes = ["nonmusic", idx_to_class[top1_idx], idx_to_class[top3_indices[1]]]
                else:
                    pred_class = idx_to_class[top1_idx]
                    pred_prob = float(p[top1_idx])
                    top3_classes = [idx_to_class[idx] for idx in top3_indices]

                meta["predicted_class"] = pred_class
                meta["confidence"] = round(pred_prob, 4)
                meta["top3_classes"] = top3_classes
                window_results.append(meta)

            batch_chunks = []
            batch_meta = []

    print(f"    Completed inference on {len(window_results)} windows.")

    # 9. Ground Truth Alignment & Metrics Computation
    print("[*] Aligning predictions with cue sheet ground truth...")
    overall_metrics, per_cue_summary, per_class_summary, window_results = align_and_compute_metrics(
        window_results, cues, labels
    )

    # Overall report dictionary
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
            "fmin": fmin,
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
        "per_cue_evaluations": per_cue_summary
    }

    if save_all_windows:
        report["detailed_window_predictions"] = window_results
    else:
        report["window_predictions_sample"] = window_results[:50]

    # Save to JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    music_metrics = overall_metrics["music_cues"]
    nonmusic_metrics = overall_metrics["nonmusic_cues"]

    print(f"\n{'='*75}")
    print("  COME DRINK WITH ME - CNN CUE SHEET EVALUATION REPORT")
    print(f"{'='*75}")
    print(f"Model Evaluated        : {model_path}")
    print(f"Total Audio Length     : {total_audio_sec/60:.2f} min ({len(window_results)} windows)")
    print(f"Music Cue Windows      : {music_metrics['windows_count']} windows across {len(cues)} cues")
    print(f"-> Top-1 Music Accuracy: {music_metrics['top1_dominant_accuracy_percent']:.2f}% ({music_metrics['top1_dominant_hits']}/{music_metrics['windows_count']} hits)")
    print(f"-> Top-3 Music Accuracy: {music_metrics['top3_dominant_accuracy_percent']:.2f}% ({music_metrics['top3_dominant_hits']}/{music_metrics['windows_count']} hits)")
    print(f"Non-Music Windows      : {nonmusic_metrics['windows_count']} windows")
    print(f"-> Non-Music Accuracy  : {nonmusic_metrics['accuracy_percent']:.2f}% ({nonmusic_metrics['correct_hits']}/{nonmusic_metrics['windows_count']} correct)")
    print(f"Total Film Accuracy    : {overall_metrics['total_accuracy_percent']:.2f}%")
    print(f"Results saved to       : {output_path}")
    print(f"{'='*75}\n")

    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate CNN against Come Drink With Me cue sheet ground truth.")
    parser.add_argument("--audio", type=str, default="data/raw/Come Drink With Me.mp4", help="Path to video or audio")
    parser.add_argument("--model", type=str, default="models/resnet50_cnn.pt", help="Path to model weights")
    parser.add_argument(
        "--cuesheet",
        type=str,
        default="data/test/come-drink-with-me/Film Music Cue Sheet for Come Drink with Me (English) - English Version.csv",
        help="Path to cue sheet CSV"
    )
    parser.add_argument(
        "--mapping",
        type=str,
        default="data/test/come-drink-with-me/instrument_mapping.json",
        help="Path to instrument mapping JSON"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/inference_eval_come_drink_with_me.json",
        help="Path to output JSON report"
    )
    parser.add_argument("--window-sec", type=float, default=4.0, help="Window length in seconds")
    parser.add_argument("--stride-sec", type=float, default=1.0, help="Stride length in seconds")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--fmin", type=float, default=65.4, help="CQT minimum frequency in Hz")
    parser.add_argument("--rms-threshold", type=float, default=0.001, help="RMS silence cutoff threshold")
    parser.add_argument("--save-all-windows", action="store_true", help="Save all window predictions in JSON")

    args = parser.parse_args()

    run_evaluation(
        audio_path=args.audio,
        model_path=args.model,
        cuesheet_path=args.cuesheet,
        mapping_path=args.mapping,
        output_path=args.output,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        batch_size=args.batch_size,
        fmin=args.fmin,
        rms_threshold=args.rms_threshold,
        save_all_windows=args.save_all_windows
    )


if __name__ == "__main__":
    main()
