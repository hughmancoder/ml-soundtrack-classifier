import os
import sys
import json
import argparse
from pathlib import Path
import soundfile as sf

# Ensure src/ is in sys.path when invoked directly as `python3 scripts/data/slice_film.py`
repo_root = Path(__file__).resolve().parents[2]
src_dir = str(repo_root / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from soundtrack_classifier.ingestion.audio_utils import extract_audio_from_video


def hhmmss_to_seconds(t_str: str) -> float:
    """Converts HH:MM:SS or MM:SS or seconds string to float seconds."""
    parts = str(t_str).strip().split(":")
    if len(parts) == 3:
        h, m, s = map(float, parts)
        return h * 3600.0 + m * 60.0 + s
    elif len(parts) == 2:
        m, s = map(float, parts)
        return m * 60.0 + s
    else:
        return float(parts[0])


def resolve_candidate_path(candidates: list) -> str:
    """Returns the first existing path from candidates, or the first non-empty element if none exist."""
    for c in candidates:
        if c and os.path.exists(c):
            return str(c)
    return str(candidates[0]) if candidates else ""


def main():
    parser = argparse.ArgumentParser(
        description="Slice film audio segments based on JSON annotations into labeled class folders (e.g. nonmusic/, percussion/, erhu/) preserving original duration and sample rate."
    )
    parser.add_argument(
        "--json-path",
        type=str,
        default=None,
        help="Path to annotations JSON file (default: searches data/raw/come_drink_with_me_annotations.json)"
    )
    parser.add_argument(
        "--audio-path",
        type=str,
        default=None,
        help="Path to source WAV audio file (default: searches data/raw/Come Drink With Me.wav)"
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default=None,
        help="Path to source video file if WAV needs extraction"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/train",
        help="Base training output directory containing class folders (default: data/train)"
    )
    args = parser.parse_args()

    # 1. Resolve JSON path
    json_candidates = [
        args.json_path,
        "data/raw/come_drink_with_me_annotations.json",
        "data/raw/come_drink_with_me.json",
        "data/annotations/come_drink_with_me.json"
    ]
    resolved_json = resolve_candidate_path([c for c in json_candidates if c])
    if not os.path.exists(resolved_json):
        print(f"Error: Annotations JSON not found. Checked: {json_candidates}")
        sys.exit(1)

    # 2. Load annotations JSON first to inspect metadata (source_file, etc.)
    with open(resolved_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    annotations = data.get("annotations", [])
    if not annotations:
        print(f"Warning: No annotations found in {resolved_json}.")
        return

    source_file = data.get("source_file", "Come Drink With Me.mp4")
    source_stem = Path(source_file).stem
    film_slug = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in source_stem.lower())
    # Short prefix for filenames (e.g., 'cdwm' for 'come_drink_with_me')
    words = [w for w in source_stem.split() if w]
    prefix = "".join(w[0].lower() for w in words) if len(words) >= 2 else film_slug

    # 3. Resolve Audio / Video paths
    audio_candidates = [
        args.audio_path,
        f"data/raw/{source_stem}.wav",
        "data/raw/Come Drink With Me.wav",
        f"data/raw/{source_file}.wav"
    ]
    resolved_audio = resolve_candidate_path([c for c in audio_candidates if c])

    video_candidates = [
        args.video_path,
        f"data/raw/{source_file}",
        f"data/raw/{source_stem}.mp4",
        f"data/raw/{source_stem}/{source_file}",
        "data/raw/Come Drink With Me.mp4",
        "data/raw/Come Drink With Me/Come Drink With Me.mp4"
    ]
    resolved_video = resolve_candidate_path([c for c in video_candidates if c])

    # Extract audio if WAV not found
    if not os.path.exists(resolved_audio):
        if os.path.exists(resolved_video):
            print(f"Extracting native audio from video {resolved_video}...")
            extract_audio_from_video(resolved_video, resolved_audio, sr=None)
        else:
            print(f"Error: Neither audio ({resolved_audio}) nor video ({resolved_video}) found.")
            sys.exit(1)

    # 4. Read audio info without loading full file into memory
    info = sf.info(resolved_audio)
    native_sr = info.samplerate
    total_duration = info.duration
    print(f"Source Film Audio: {resolved_audio}")
    print(f"Native Sample Rate: {native_sr} Hz, Total Duration: {total_duration:.2f}s ({total_duration/60:.1f} min)")
    print(f"Base Output Directory: {args.output_dir}")
    print(f"Found {len(annotations)} annotated film segments to slice into target class folders...\n")

    saved_by_class = {}

    for idx, ann in enumerate(annotations, 1):
        t_start_str = ann.get("timestamp_start", "00:00:00")
        t_end_str = ann.get("timestamp_end", "00:00:00")
        label = str(ann.get("label", "nonmusic")).strip().lower()
        desc = str(ann.get("description", label)).strip()
        clean_desc = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in desc)

        start_sec = hhmmss_to_seconds(t_start_str)
        end_sec = hhmmss_to_seconds(t_end_str)
        duration_sec = end_sec - start_sec

        if duration_sec <= 0:
            print(f"Skipping invalid duration [{t_start_str} -> {t_end_str}]")
            continue

        # Target class folder: e.g. data/train/nonmusic or data/train/percussion
        target_folder = os.path.join(args.output_dir, label)
        os.makedirs(target_folder, exist_ok=True)

        start_frame = int(round(start_sec * native_sr))
        num_frames = int(round(duration_sec * native_sr))

        # Direct seeking read - fast and memory-efficient
        chunk, _ = sf.read(resolved_audio, start=start_frame, frames=num_frames, dtype="float32")

        # Convert to mono if multi-channel
        if chunk.ndim > 1:
            chunk = chunk.mean(axis=-1)

        # Generate descriptive filename
        t_start_clean = t_start_str.replace(":", "")
        t_end_clean = t_end_str.replace(":", "")
        filename = f"{prefix}_{label}_{idx:02d}_{clean_desc}_{t_start_clean}_{t_end_clean}.wav"
        filepath = os.path.join(target_folder, filename)

        sf.write(filepath, chunk, native_sr)

        saved_by_class.setdefault(label, []).append((filename, duration_sec))
        print(f"[{idx}/{len(annotations)}] Saved: {label}/{filename} ({duration_sec:.2f}s, {native_sr} Hz)")

    print("\n" + "=" * 60)
    print("Film Slicing Summary:")
    print("-" * 60)
    total_clips = 0
    for cls_name, clips in saved_by_class.items():
        total_sec = sum(d for _, d in clips)
        print(f"  - {cls_name:<15}: {len(clips):>3} clips ({total_sec:.1f}s total) -> {os.path.join(args.output_dir, cls_name)}")
        total_clips += len(clips)
    print(f"Total: {total_clips} clips written across {len(saved_by_class)} classes.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
