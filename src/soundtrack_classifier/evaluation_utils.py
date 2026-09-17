import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import soundfile as sf

from soundtrack_classifier.audio_utils import extract_audio_from_video


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


def load_cue_sheet(csv_path: str, mapping: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parses the cue sheet CSV and maps raw instrument descriptions
    to target training classes using the provided mapping dictionary.
    Handles both nested {'instrument_mapping': {...}} and flat dictionaries.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cue sheet CSV not found: {csv_path}")

    # Ensure flat dictionary mapping
    inst_mapping = mapping.get("instrument_mapping", mapping)

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
                if key in inst_mapping:
                    mapped_classes.add(inst_mapping[key])

            # Also check gong and drum section if any instruments are noted there
            if gong_drum:
                for item in [x.strip().rstrip(".") for x in gong_drum.replace(";", ",").split(",") if x.strip()]:
                    if item.lower() in inst_mapping:
                        mapped_classes.add(inst_mapping[item.lower()])

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


def prepare_audio_file(audio_input_path: str, target_sr: Optional[int] = None) -> str:
    """
    Ensures input audio is available as a .wav file at target_sr (if specified).
    If given a video file, extracts to a .wav counterpart. If target_sr is provided,
    suffixes the filename to prevent clashing (unless it's standard 44.1kHz behavior).
    """
    video_exts = ('.mp4', '.mkv', '.mov', '.avi', '.m4v', '.webm', '.flv')
    input_path = Path(audio_input_path)

    if input_path.suffix.lower() in video_exts:
        if target_sr:
            # E.g., for CRNN which uses 22050
            wav_path = input_path.with_name(f"{input_path.stem}_{target_sr}.wav")
            if not wav_path.exists():
                # Check if standard .wav exists and matches samplerate
                std_wav = input_path.with_suffix('.wav')
                if std_wav.exists():
                    info = sf.info(str(std_wav))
                    if info.samplerate == target_sr:
                        return str(std_wav)
                print(f"Extracting audio track from video: {audio_input_path} -> {wav_path} at {target_sr} Hz...")
                extract_audio_from_video(str(input_path), str(wav_path), sr=target_sr)
            return str(wav_path)
        else:
            # Default behavior (e.g., CNN or AST where extraction handles sample rate downstream)
            wav_path = input_path.with_suffix('.wav')
            if not wav_path.exists():
                print(f"Extracting audio track from video: {audio_input_path} -> {wav_path}...")
                extract_audio_from_video(str(input_path), str(wav_path), sr=44100)
            return str(wav_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_input_path}")

    return str(input_path)


def align_and_compute_metrics(
    window_results: List[Dict[str, Any]],
    cues: List[Dict[str, Any]],
    labels: List[str]
) -> tuple:
    """
    Aligns window predictions with cue sheet ground truth and computes evaluation metrics.

    Returns:
        tuple: (aggregate_metrics_dict, per_cue_summary, per_class_summary, window_results)
    """
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
        top3 = win.get("top3_classes", [pred])
        class_stats[pred]["predicted_windows"] += 1

        # Check if window falls into any cue
        matching_cues = [
            c_eval for c_eval in cue_evals
            if c_eval["start_sec"] <= mid_sec <= c_eval["end_sec"]
        ]

        if matching_cues:
            music_windows_count += 1
            # Merge ground truth of matching cues
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
    total_windows = len(window_results)
    music_top1_acc = (music_top1_hits / music_windows_count * 100.0) if music_windows_count > 0 else 0.0
    music_top3_acc = (music_top3_hits / music_windows_count * 100.0) if music_windows_count > 0 else 0.0
    nonmusic_acc = (nonmusic_hits / nonmusic_windows_count * 100.0) if nonmusic_windows_count > 0 else 0.0
    total_correct = music_top1_hits + nonmusic_hits
    total_acc = (total_correct / max(1, total_windows) * 100.0) if total_windows > 0 else 0.0

    metrics = {
        "total_windows_analyzed": total_windows,
        "total_accuracy_percent": round(total_acc, 2),
        "music_cues": {
            "total_windows": music_windows_count,
            "top1_dominant_accuracy_percent": round(music_top1_acc, 2),
            "top3_dominant_accuracy_percent": round(music_top3_acc, 2)
        },
        "nonmusic_cues": {
            "total_windows": nonmusic_windows_count,
            "accuracy_percent": round(nonmusic_acc, 2)
        }
    }

    # Format per-cue evaluations
    per_cue_summary = []
    for c in cue_evals:
        n_win = c["total_windows"]
        cue_top1_acc = (c["top1_hits"] / n_win * 100.0) if n_win > 0 else 0.0
        cue_top3_acc = (c["top3_hits"] / n_win * 100.0) if n_win > 0 else 0.0

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

    return metrics, per_cue_summary, per_class_summary, window_results
