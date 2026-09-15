"""
Test CNN Model Against Film Cue Sheet Ground Truth Annotations.

Runs sliding-window inference across film audio, maps predictions against
timestamped cue sheet ground truth, calculates polyphonic dominant-hit accuracy,
and saves detailed evaluation results to a JSON log file.
"""

import os
import sys
import json
import csv
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any

import torch
import numpy as np
import soundfile as sf
from nnAudio.features.cqt import CQT1992v2

from soundtrack_classifier.cnn.resnet import ResNet18Classifier
from soundtrack_classifier.ingestion.audio_utils import extract_audio_from_video


def parse_timestamp(t_str: str) -> float:
    """Parses HH:MM:SS or MM:SS format into total seconds."""
    parts = t_str.strip().split(":")
    if len(parts) == 3:
        h, m, s = map(float, parts)
        return h * 3600.0 + m * 60.0 + s
    elif len(parts) == 2:
        m, s = map(float, parts)
        return m * 60.0 + s
    return float(parts[0])


def load_cue_sheet(
    csv_path: str, 
    mapping: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    Parses the cue sheet CSV and maps raw instrument descriptions
    to target training classes using the provided mapping dictionary.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cue sheet CSV not found: {csv_path}")

    cues = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            t_period = row.get("Time period", "")
            if not t_period or "-" not in t_period:
                continue

            start_str, end_str = t_period.split("-")
            start_sec = parse_timestamp(start_str)
            end_sec = parse_timestamp(end_str)

            plot = row.get("Plot Description", "").strip()
            genre = row.get("Music Genre: (Chinese Music/Western Music/Electronic Music)", "").strip()
            rec = row.get("Musical Instrument Recognition", "").strip()
            gong_drum = row.get("Gong and drum section", "").strip()

            # Split instrument text by comma / semicolon
            raw_instruments = [
                item.strip().rstrip(".") 
                for item in rec.replace(";", ",").split(",") 
                if item.strip()
            ]

            mapped_classes = set()
            for item in raw_instruments:
                key = item.lower()
                if key in mapping:
                    mapped_classes.add(mapping[key])

            # Also check gong and drum section if any instruments are noted there
            if gong_drum:
                for item in [x.strip().rstrip(".") for x in gong_drum.replace(";", ",").split(",") if x.strip()]:
                    if item.lower() in mapping:
                        mapped_classes.add(mapping[item.lower()])

            cues.append({
                "cue_id": idx + 1,
                "time_period": t_period.strip(),
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": end_sec - start_sec,
                "plot_description": plot,
                "music_genre": genre,
                "raw_instruments": raw_instruments,
                "ground_truth_classes": sorted(list(mapped_classes))
            })

    return cues


def prepare_audio_file(audio_input_path: str, target_sr: int = 44100) -> str:
    """
    Ensures input audio is available as a .wav file.
    If given a video file (.mp4, .mkv), extracts to a .wav counterpart.
    """
    video_exts = ('.mp4', '.mkv', '.mov', '.avi', '.m4v', '.webm', '.flv')
    input_path = Path(audio_input_path)

    if input_path.suffix.lower() in video_exts:
        wav_path = input_path.with_suffix('.wav')
        if not wav_path.exists():
            print(f"Extracting audio track from video: {audio_input_path} -> {wav_path}...")
            extract_audio_from_video(str(input_path), str(wav_path), sr=target_sr)
        return str(wav_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_input_path}")

    return str(input_path)


def run_evaluation(
    audio_path: str,
    model_path: str,
    cuesheet_path: str,
    mapping_path: str,
    output_path: str,
    window_sec: float = 3.0,
    stride_sec: float = 1.0,
    batch_size: int = 64,
    fmin: float = 32.7,
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
        "sheng", "dizi", "erhu", "pipa", "suona", "guzheng", 
        "yangqin", "percussion", "strings", "woodwind", "brass", "nonmusic"
    ]
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict", checkpoint)
        if "classes" in checkpoint:
            labels = checkpoint["classes"]
    else:
        state_dict = checkpoint

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
    
    # Initialize per-cue tracking
    cue_evals = []
    for cue in cues:
        cue_evals.append({
            **cue,
            "total_windows": 0,
            "top1_hits": 0,
            "top3_hits": 0,
            "predictions_count": {},
            "windows": []
        })

    music_windows_count = 0
    music_top1_hits = 0
    music_top3_hits = 0

    nonmusic_windows_count = 0
    nonmusic_hits = 0

    # Per-class metrics
    class_stats = {c: {"gt_active_windows": 0, "predicted_windows": 0, "hits_when_active": 0} for c in labels}

    for win in window_results:
        mid_sec = win["mid_sec"]
        pred = win["predicted_class"]
        top3 = win["top3_classes"]
        class_stats[pred]["predicted_windows"] += 1

        # Check if window falls into any cue
        matching_cues = [
            c_eval for c_eval in cue_evals 
            if c_eval["start_sec"] <= mid_sec <= c_eval["end_sec"]
        ]

        if matching_cues:
            music_windows_count += 1
            # Merge ground truth of matching cues (in case of adjacent/overlapping boundaries)
            combined_gt = set()
            for mc in matching_cues:
                combined_gt.update(mc["ground_truth_classes"])
                mc["total_windows"] += 1
                mc["predictions_count"][pred] = mc["predictions_count"].get(pred, 0) + 1

            for c in combined_gt:
                if c in class_stats:
                    class_stats[c]["gt_active_windows"] += 1

            # Top-1 Dominant Hit Check: Is predicted dominant instrument active in cue?
            is_hit = pred in combined_gt
            is_top3_hit = any(c in combined_gt for c in top3)

            if is_hit:
                music_top1_hits += 1
                class_stats[pred]["hits_when_active"] += 1
                for mc in matching_cues:
                    mc["top1_hits"] += 1

            if is_top3_hit:
                music_top3_hits += 1
                for mc in matching_cues:
                    mc["top3_hits"] += 1

            win["ground_truth_classes"] = sorted(list(combined_gt))
            win["is_music_cue"] = True
            win["is_hit"] = is_hit
            win["is_top3_hit"] = is_top3_hit
        else:
            nonmusic_windows_count += 1
            is_nonmusic_hit = (pred == "nonmusic")
            if is_nonmusic_hit:
                nonmusic_hits += 1

            win["ground_truth_classes"] = ["nonmusic"]
            win["is_music_cue"] = False
            win["is_hit"] = is_nonmusic_hit
            win["is_top3_hit"] = "nonmusic" in top3

    # Calculate final aggregate metrics
    music_top1_acc = (music_top1_hits / music_windows_count * 100.0) if music_windows_count > 0 else 0.0
    music_top3_acc = (music_top3_hits / music_windows_count * 100.0) if music_windows_count > 0 else 0.0
    nonmusic_acc = (nonmusic_hits / nonmusic_windows_count * 100.0) if nonmusic_windows_count > 0 else 0.0
    total_correct = music_top1_hits + nonmusic_hits
    total_acc = (total_correct / total_windows * 100.0) if total_windows > 0 else 0.0

    # Format per-cue evaluations
    per_cue_summary = []
    for c in cue_evals:
        n_win = c["total_windows"]
        cue_top1_acc = (c["top1_hits"] / n_win * 100.0) if n_win > 0 else 0.0
        cue_top3_acc = (c["top3_hits"] / n_win * 100.0) if n_win > 0 else 0.0
        
        # Sort predictions by frequency
        sorted_preds = sorted(c["predictions_count"].items(), key=lambda x: x[1], reverse=True)

        per_cue_summary.append({
            "cue_id": c["cue_id"],
            "time_period": c["time_period"],
            "start_sec": c["start_sec"],
            "end_sec": c["end_sec"],
            "duration_sec": c["duration_sec"],
            "plot_description": c["plot_description"],
            "ground_truth_classes": c["ground_truth_classes"],
            "total_windows": n_win,
            "top1_hits": c["top1_hits"],
            "top1_accuracy_percent": round(cue_top1_acc, 2),
            "top3_hits": c["top3_hits"],
            "top3_accuracy_percent": round(cue_top3_acc, 2),
            "prediction_distribution": dict(sorted_preds)
        })

    # Format per-class summary
    per_class_summary = {}
    for c, stat in class_stats.items():
        gt_active = stat["gt_active_windows"]
        predicted = stat["predicted_windows"]
        hits = stat["hits_when_active"]
        precision = (hits / predicted * 100.0) if predicted > 0 else 0.0
        recall = (hits / gt_active * 100.0) if gt_active > 0 else 0.0
        per_class_summary[c] = {
            "gt_active_windows": gt_active,
            "predicted_windows": predicted,
            "hits_when_active": hits,
            "precision_percent": round(precision, 2),
            "recall_detection_percent": round(recall, 2)
        }

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
            "total_windows": total_windows
        },
        "overall_metrics": {
            "music_cues": {
                "total_windows": music_windows_count,
                "top1_hits": music_top1_hits,
                "top1_accuracy_percent": round(music_top1_acc, 2),
                "top3_hits": music_top3_hits,
                "top3_accuracy_percent": round(music_top3_acc, 2)
            },
            "non_music": {
                "total_windows": nonmusic_windows_count,
                "correct_rejections": nonmusic_hits,
                "accuracy_percent": round(nonmusic_acc, 2)
            },
            "total_film": {
                "total_windows": total_windows,
                "total_correct": total_correct,
                "overall_accuracy_percent": round(total_acc, 2)
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

    print(f"\n{'='*75}")
    print(f"  COME DRINK WITH ME - CNN CUE SHEET EVALUATION REPORT")
    print(f"{'='*75}")
    print(f"Model Evaluated        : {model_path}")
    print(f"Total Audio Length     : {total_audio_sec/60:.2f} min ({total_windows} windows)")
    print(f"Music Cue Windows      : {music_windows_count} windows across {len(cues)} cues")
    print(f"-> Top-1 Music Accuracy: {music_top1_acc:.2f}% ({music_top1_hits}/{music_windows_count} hits)")
    print(f"-> Top-3 Music Accuracy: {music_top3_acc:.2f}% ({music_top3_hits}/{music_windows_count} hits)")
    print(f"Non-Music Windows      : {nonmusic_windows_count} windows")
    print(f"-> Non-Music Accuracy  : {nonmusic_acc:.2f}% ({nonmusic_hits}/{nonmusic_windows_count} correct)")
    print(f"Total Film Accuracy    : {total_acc:.2f}%")
    print(f"Results saved to       : {output_path}")
    print(f"{'='*75}\n")

    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate CNN against Come Drink With Me cue sheet ground truth.")
    parser.add_argument("--audio", type=str, default="data/raw/Come Drink With Me.mp4", help="Path to video or audio")
    parser.add_argument("--model", type=str, default="models/resnet18_cnn_best.pt", help="Path to model weights")
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
    parser.add_argument("--window-sec", type=float, default=3.0, help="Window length in seconds")
    parser.add_argument("--stride-sec", type=float, default=1.0, help="Stride length in seconds")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--fmin", type=float, default=32.7, help="CQT minimum frequency in Hz")
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
