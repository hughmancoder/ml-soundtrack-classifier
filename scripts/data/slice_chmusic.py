import os
import sys
import glob
import json
import shutil
import argparse
from pathlib import Path

# Ensure src/ is in sys.path when invoked directly as `python3 scripts/slice_chmusic.py`
repo_root = Path(__file__).resolve().parent.parent
src_dir = str(repo_root / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from soundtrack_classifier.cnn.dataset import slice_audio_file

# Optional hjson import with fallback to standard json
try:
    import hjson
except ImportError:
    hjson = None


def load_chmusic_mapping(map_path: str = "config/chmusic_class_map.hjson") -> dict:
    """
    Loads ChMusic file-to-class mapping from an HJSON or JSON file.
    Searches config/chmusic_class_map.hjson, config/chmusic_class_map.json,
    or data/raw/ChMusic/ counterparts.
    """
    candidates = [
        map_path,
        "config/chmusic_class_map.hjson",
        "config/chmusic_class_map.json",
        "data/raw/ChMusic/chmusic_class_map.hjson",
        "data/raw/ChMusic/chmusic_class_map.json",
        "chmusic_class_map.hjson",
        "chmusic_class_map.json"
    ]
    resolved_path = None
    for c in candidates:
        if os.path.exists(c):
            resolved_path = c
            break

    if not resolved_path:
        raise FileNotFoundError(f"Could not locate chmusic class mapping file. Checked: {candidates}")

    with open(resolved_path, "r", encoding="utf-8") as f:
        if resolved_path.endswith(".hjson"):
            if hjson is not None:
                return hjson.load(f)
            # If hjson package is not installed, fallback to .json equivalent if exists
            json_equiv = resolved_path.replace(".hjson", ".json")
            if os.path.exists(json_equiv):
                with open(json_equiv, "r", encoding="utf-8") as jf:
                    return json.load(jf)
            raise ImportError(
                "hjson module is required to parse .hjson files. "
                "Run 'pip install hjson' or use 'config/chmusic_class_map.json'."
            )
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Partition ChMusic dataset into class folders in train/ keeping original length and sample rate."
    )
    parser.add_argument("--map-file", type=str, default="config/chmusic_class_map.hjson", help="Path to chmusic_class_map.hjson or .json")
    parser.add_argument("--input-dir", type=str, default="data/raw/ChMusic/Musics", help="Directory containing raw ChMusic .wav files")
    parser.add_argument("--output-dir", type=str, default="data/train", help="Target train directory (default: data/train)")
    parser.add_argument("--slice", action="store_true", help="Slice into fixed duration chunks instead of keeping original length")
    parser.add_argument("--duration-sec", type=float, default=3.0, help="Slice duration in seconds if --slice is enabled (default: 3.0s)")
    parser.add_argument("--sr", type=int, default=44100, help="Target sample rate (default: 44100)")
    parser.add_argument("--rms-threshold", type=float, default=0.005, help="RMS energy threshold to discard silence if --slice is enabled")
    args = parser.parse_args()

    mapping_data = load_chmusic_mapping(args.map_file)
    train_labels = mapping_data.get("train_labels", [])
    files_map = mapping_data.get("files", {})
    instruments_map = mapping_data.get("instruments", {})

    input_dir = args.input_dir
    output_base_dir = args.output_dir

    if not os.path.exists(input_dir):
        print(f"Error: {input_dir} not found. Please ensure ChMusic dataset is downloaded.")
        return

    # Ensure all target class folders exist in output directory
    for label in train_labels:
        os.makedirs(os.path.join(output_base_dir, label), exist_ok=True)

    wav_files = sorted(glob.glob(os.path.join(input_dir, "*.wav")))
    if not wav_files:
        print(f"No .wav files found in {input_dir}.")
        return

    print(f"Found {len(wav_files)} audio files in {input_dir}.")
    print(f"Target directory: {output_base_dir}")
    if args.slice:
        print(f"Mode: Slicing into {args.duration_sec}s chunks at {args.sr} Hz")
    else:
        print(f"Mode: Keeping original length and original sample rate (bit-for-bit)")

    total_processed = 0
    total_outputs = 0

    for file_path in wav_files:
        filename = Path(file_path).name
        target_class = None
        music_name = None

        if filename in files_map:
            meta = files_map[filename]
            target_class = meta.get("target_class")
            music_name = meta.get("music_name")
        else:
            # Fallback by prefix (e.g. 1.1.wav -> prefix 1)
            prefix = filename.split(".")[0]
            if prefix in instruments_map:
                target_class = instruments_map[prefix].get("target_class")

        if not target_class:
            print(f"Skipping {filename}: No target class mapping found.")
            continue

        target_folder = os.path.join(output_base_dir, target_class)
        os.makedirs(target_folder, exist_ok=True)

        if args.slice:
            num_slices = slice_audio_file(
                input_path=file_path,
                output_dir=target_folder,
                duration_sec=args.duration_sec,
                sr=args.sr,
                rms_threshold=args.rms_threshold
            )
            total_outputs += num_slices
        else:
            dest_file = os.path.join(target_folder, filename)
            shutil.copy2(file_path, dest_file)
            total_outputs += 1
            desc = f" - '{music_name}'" if music_name else ""
            print(f"Copied {filename}{desc} -> {target_class}/{filename}")

        total_processed += 1

    mode_str = f"{total_outputs} chunks" if args.slice else f"{total_outputs} original full-length tracks"
    print(f"\nDone! Processed {total_processed} ChMusic files -> Output {mode_str} across classes in '{output_base_dir}'.")


if __name__ == "__main__":
    main()
