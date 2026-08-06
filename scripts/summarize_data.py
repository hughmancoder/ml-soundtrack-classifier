import os
import glob
import numpy as np
import soundfile as sf
import yaml

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def summarize_data(data_dir: str, silence_threshold: float = 0.005, output_dir: str = "data"):
    """
    Summarizes the dataset distribution in data_dir.
    Checks class counts and identifies silent or near-silent files.
    Writes silent clip paths to {output_dir}/silent_clips.txt.
    """
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' does not exist.")
        return

    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    classes.sort()

    total_files = 0
    total_silent = 0
    silent_paths = []  # collect all silent file paths
    
    print("=" * 60)
    print(f"{'Instrument Class':<20} | {'Total Files':<12} | {'Silent Files':<12}")
    print("=" * 60)

    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        files = glob.glob(os.path.join(cls_dir, "*.wav"))
        
        class_total = len(files)
        class_silent = 0
        
        for f in files:
            try:
                # Use soundfile for fast reading
                audio, _ = sf.read(f)
                
                # Check for silence using RMS energy
                rms_energy = np.sqrt(np.mean(audio**2))
                if rms_energy < silence_threshold:
                    class_silent += 1
                    silent_paths.append(f)
            except Exception as e:
                print(f"Error reading file {f}: {e}")
                
        print(f"{cls:<20} | {class_total:<12} | {class_silent:<12}")
        
        total_files += class_total
        total_silent += class_silent

    print("=" * 60)
    print(f"{'TOTAL':<20} | {total_files:<12} | {total_silent:<12}")
    print("=" * 60)
    
    # Write silent clips list to data/silent_clips.txt
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "silent_clips.txt")
    with open(output_path, "w") as out:
        out.write(f"# Silent clips (RMS < {silence_threshold})\n")
        out.write(f"# Total: {total_silent} / {total_files}\n\n")
        for path in sorted(silent_paths):
            out.write(path + "\n")
    print(f"\nSilent clips list written to: {output_path}")

    if total_silent > 0:
        print(f"Warning: Found {total_silent} silent or near-silent files across the dataset.")
        print(f"Consider adjusting your slicing script or reviewing the raw audio.")
    else:
        print("All files appear to contain valid audio data (No silent files detected).")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Summarize training dataset distribution.")
    parser.add_argument("--data_dir", type=str, default="data/train", help="Path to the training data directory.")
    parser.add_argument("--threshold", type=float, default=0.005, help="RMS threshold below which a file is considered silent.")
    args = parser.parse_args()
    
    # Try to load from config if data_dir is default
    if args.data_dir == "data/train" and os.path.exists("config/config.yaml"):
        config = load_config()
        if 'paths' in config and 'data_dir' in config['paths']:
            args.data_dir = config['paths']['data_dir']
            
    summarize_data(args.data_dir, silence_threshold=args.threshold)
