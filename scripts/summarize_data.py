import os
import glob
import numpy as np
import soundfile as sf
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def format_time(seconds: float) -> str:
    """Formats duration into HH:MM:SS or MM:SS string."""
    total_sec = int(round(seconds))
    hours = total_sec // 3600
    minutes = (total_sec % 3600) // 60
    sec = total_sec % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def summarize_data(data_dir: str, silence_threshold: float = 0.005, output_dir: str = "data"):
    """
    Summarizes the dataset distribution in data_dir:
    - Counts files per class
    - Computes total duration in seconds (and HH:MM:SS) per class and total
    - Checks for silent or near-silent files using RMS energy
    - Writes silent clip paths to {output_dir}/silent_clips.txt
    """
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' does not exist.")
        return

    # Check classes defined in directory or config
    config = load_config()
    cfg_labels = config.get("train_labels", [])
    dir_classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith(".")]
    
    # Preserve train_labels order, append any additional discovered directories
    classes = [c for c in cfg_labels if c in dir_classes]
    for c in sorted(dir_classes):
        if c not in classes:
            classes.append(c)

    total_files = 0
    total_duration_sec = 0.0
    total_silent = 0
    silent_paths = []

    print("=" * 78)
    print(f"{'Instrument Class':<18} | {'Total Files':<11} | {'Duration (s)':<14} | {'Time (m:s)':<10} | {'Silent Files':<10}")
    print("=" * 78)

    audio_extensions = ("*.wav", "*.flac", "*.mp3", "*.ogg")

    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        files = []
        for ext in audio_extensions:
            files.extend(glob.glob(os.path.join(cls_dir, ext)))
        files.sort()

        class_total = len(files)
        class_duration = 0.0
        class_silent = 0

        for f in files:
            try:
                info = sf.info(f)
                class_duration += info.duration

                # Calculate RMS energy for silence check
                audio, _ = sf.read(f, dtype="float32", always_2d=False)
                rms_energy = np.sqrt(np.mean(audio**2))
                if rms_energy < silence_threshold:
                    class_silent += 1
                    silent_paths.append(f)
            except Exception as e:
                print(f"Error reading file {f}: {e}")

        total_files += class_total
        total_duration_sec += class_duration
        total_silent += class_silent

        dur_sec_str = f"{class_duration:.1f}s"
        dur_time_str = format_time(class_duration)
        print(f"{cls:<18} | {class_total:<11} | {dur_sec_str:<14} | {dur_time_str:<10} | {class_silent:<10}")

    print("=" * 78)
    total_sec_str = f"{total_duration_sec:.1f}s"
    total_time_str = format_time(total_duration_sec)
    print(f"{'TOTAL':<18} | {total_files:<11} | {total_sec_str:<14} | {total_time_str:<10} | {total_silent:<10}")
    print("=" * 78)

    # Summary text
    total_hrs = total_duration_sec / 3600.0
    print(f"\n[+] Total Training Data Duration: {total_duration_sec:.2f} seconds (~{total_hrs:.2f} hours / {total_time_str})")

    # Write silent clips list to data/silent_clips.txt
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "silent_clips.txt")
    with open(output_path, "w", encoding="utf-8") as out:
        out.write(f"# Silent clips (RMS < {silence_threshold})\n")
        out.write(f"# Total: {total_silent} / {total_files}\n")
        out.write(f"# Dataset Total Duration: {total_duration_sec:.2f}s ({total_time_str})\n\n")
        for path in sorted(silent_paths):
            out.write(path + "\n")

    print(f"[+] Silent clips report written to: {output_path}")

    if total_silent > 0:
        print(f"[!] Warning: Found {total_silent} silent or near-silent files across the dataset.")
    else:
        print("[+] All files appear to contain valid audio data (No silent files detected).")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Summarize training dataset distribution and duration.")
    parser.add_argument("--data_dir", type=str, default="data/train", help="Path to the training data directory.")
    parser.add_argument("--threshold", type=float, default=0.005, help="RMS threshold below which a file is considered silent.")
    args = parser.parse_args()

    # Try to load from config if data_dir is default
    if args.data_dir == "data/train" and os.path.exists("config/config.yaml"):
        config = load_config()
        if "paths" in config and "data_dir" in config["paths"]:
            args.data_dir = config["paths"]["data_dir"]

    summarize_data(args.data_dir, silence_threshold=args.threshold)
