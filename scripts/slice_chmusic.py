import os
import glob
from pathlib import Path
from soundtrack_classifier.cnn.dataset import slice_audio_file

# Mapping from ChMusic prefix number to our model classes
CHMUSIC_MAPPING = {
    "1": "erhu",
    "2": "pipa",
    "3": "strings",    # Sanxian (plucked string)
    "4": "dizi",
    "5": "suona",
    "6": "strings",    # Zhuiqin (bowed string)
    "7": "strings",    # Zhongruan (plucked string)
    "8": "strings",    # Liuqin (plucked string)
    "9": "guzheng",
    "10": "yangqin",
    "11": "sheng"
}

def main():
    input_dir = "data/raw/ChMusic/Musics"
    output_base_dir = "data/train"
    
    if not os.path.exists(input_dir):
        print(f"Error: {input_dir} not found. Please ensure ChMusic dataset is downloaded.")
        return

    # Find all wav files
    wav_files = glob.glob(os.path.join(input_dir, "*.wav"))
    if not wav_files:
        print(f"No .wav files found in {input_dir}.")
        return

    print(f"Found {len(wav_files)} files in ChMusic dataset. Starting slicing...")
    
    total_sliced = 0
    
    for file_path in wav_files:
        filename = Path(file_path).name
        # Filenames are like "1.1.wav" -> prefix is "1"
        prefix = filename.split(".")[0]
        
        if prefix in CHMUSIC_MAPPING:
            target_class = CHMUSIC_MAPPING[prefix]
            output_dir = os.path.join(output_base_dir, target_class)
            
            print(f"Processing {filename} -> {target_class}")
            # Use the existing slicing function from dataset.py
            # 3.0 seconds, 44100 Hz, discarding silence via RMS
            num_slices = slice_audio_file(
                input_path=file_path,
                output_dir=output_dir,
                duration_sec=3.0,
                sr=44100,
                rms_threshold=0.005 # Using a slightly lower threshold just in case
            )
            total_sliced += num_slices
        else:
            print(f"Skipping {filename}: Prefix '{prefix}' not in mapping.")

    print(f"\nDone! Generated {total_sliced} valid 3-second clips from ChMusic dataset.")

if __name__ == "__main__":
    main()
