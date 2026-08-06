import os
import json
import librosa
import soundfile as sf
import yaml
from pathlib import Path
from soundtrack_classifier.ingestion.audio_utils import extract_audio_from_video

def hhmmss_to_seconds(t_str: str) -> float:
    parts = t_str.split(':')
    if len(parts) == 3:
        h, m, s = map(int, parts)
        return h * 3600.0 + m * 60.0 + s
    elif len(parts) == 2:
        m, s = map(int, parts)
        return m * 60.0 + s
    else:
        return float(parts[0])

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    sr = config['audio']['sample_rate']
    target_duration = config['audio']['duration_sec']  # 3.0 seconds
    
    video_path = "data/raw/Come Drink With Me.mp4"
    temp_wav_path = "data/raw/Come Drink With Me.wav"
    json_path = "data/raw/come_drink_with_me.json"
    output_dir = "data/train/nonmusic"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Extract audio if not already done
    if not os.path.exists(temp_wav_path):
        print("Extracting audio from video...")
        extract_audio_from_video(video_path, temp_wav_path, sr=sr)
    else:
        print("Using existing extracted audio WAV.")
        
    # 2. Load the extracted audio
    print(f"Loading full audio from {temp_wav_path}...")
    audio, _ = librosa.load(temp_wav_path, sr=sr, mono=True)
    
    # 3. Load annotations
    with open(json_path, "r") as f:
        data = json.load(f)
        
    annotations = data.get("annotations", [])
    
    chunk_idx = 0
    
    for ann in annotations:
        start_time = hhmmss_to_seconds(ann["timestamp_start"])
        end_time = hhmmss_to_seconds(ann["timestamp_end"])
        description = ann.get("description", "nonmusic")
        
        print(f"Processing segment: {ann['timestamp_start']} to {ann['timestamp_end']} ({description})")
        
        # Convert start/end times to sample indices
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        # Slice segment
        segment = audio[start_sample:end_sample]
        
        # Slicing the segment into target_duration (3s) chunks
        chunk_samples = int(target_duration * sr)
        num_chunks = len(segment) // chunk_samples
        
        for i in range(num_chunks):
            c_start = i * chunk_samples
            c_end = c_start + chunk_samples
            chunk = segment[c_start:c_end]
            
            # Save chunk
            chunk_filename = f"cdwm_nonmusic_{description}_{chunk_idx:04d}.wav"
            chunk_filepath = os.path.join(output_dir, chunk_filename)
            sf.write(chunk_filepath, chunk, sr)
            
            chunk_idx += 1
            
    print(f"Successfully generated {chunk_idx} 3-second non-music clips in {output_dir}.")

if __name__ == "__main__":
    main()
