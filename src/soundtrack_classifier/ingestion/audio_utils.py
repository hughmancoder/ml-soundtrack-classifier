import os
import subprocess
import librosa
import soundfile as sf
import numpy as np

def extract_audio_from_video(video_path: str, output_audio_path: str, sr: int = 44100):
    """
    Extracts audio from a legacy film file (e.g., .mp4, .mkv) and saves it as a .wav file.
    Uses ffmpeg via subprocess to handle the extraction.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    print(f"Extracting audio from {video_path} to {output_audio_path}...")
    
    # We use subprocess with ffmpeg directly. We could use ffmpeg-python, 
    # but subprocess is often cleaner for a simple extraction.
    command = [
        "ffmpeg",
        "-y",               # Overwrite output
        "-i", video_path,   # Input file
        "-vn",              # Disable video
        "-acodec", "pcm_s16le", # PCM 16-bit
        "-ar", str(sr),     # Sample rate
        "-ac", "1",         # Mono (or 2 for stereo)
        output_audio_path
    ]
    
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("Extraction complete.")

def load_and_resample(audio_path: str, target_sr: int = 44100) -> tuple[np.ndarray, int]:
    """
    Loads an audio file and resamples it to the target sample rate.
    """
    print(f"Loading and resampling {audio_path} to {target_sr} Hz...")
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    return audio, sr

def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normalizes the amplitude of the audio array.
    """
    print("Normalizing audio amplitude...")
    return librosa.util.normalize(audio)

def save_audio(audio: np.ndarray, sr: int, output_path: str):
    """
    Saves the processed audio array back to disk as a .wav file.
    """
    sf.write(output_path, audio, sr)
    print(f"Saved processed audio to {output_path}")
