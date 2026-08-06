"""
Script to slice, resample, and standardize Beijing Opera Percussion (or any raw audio dataset)
into uniform 3-second, 44.1 kHz (44,100 Hz) audio clips mapped to a target output folder.

Usage:
    python scripts/slice_percussion.py
    python scripts/slice_percussion.py --input_dir data/raw/beijing_opera_percussion_instrument_1.0 --output_dir data/percussion
    python scripts/slice_percussion.py --also_train
"""

import os
import glob
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
import scipy.signal


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resamples mono audio array from orig_sr to target_sr using polyphase filtering.
    """
    if orig_sr == target_sr:
        return audio
    gcd = int(np.gcd(orig_sr, target_sr))
    up = target_sr // gcd
    down = orig_sr // gcd
    return scipy.signal.resample_poly(audio, up, down)


def process_and_slice_audio(
    input_dir: str,
    output_dir: str,
    target_sr: int = 44100,
    duration_sec: float = 3.0,
    rms_threshold: float = 0.002,
    also_train: bool = True,
):
    """
    Reads all .wav files from input_dir, resamples them to target_sr (44.1 kHz),
    slices or zero-pads them into duration_sec clips, and saves to output_dir.
    """
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return

    os.makedirs(output_dir, exist_ok=True)

    wav_files = glob.glob(os.path.join(input_dir, "*.wav"))
    wav_files.sort()

    if not wav_files:
        print(f"No .wav files found in '{input_dir}'.")
        return

    print("=" * 65)
    print(f"Processing Beijing Opera Percussion Dataset")
    print(f"Input Directory : {input_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Target Specs    : {duration_sec}s @ {target_sr} Hz (Mono)")
    print(f"RMS Threshold   : {rms_threshold}")
    print("=" * 65)

    target_samples = int(duration_sec * target_sr)
    total_input_files = len(wav_files)
    saved_clips = 0
    skipped_silent = 0

    for idx, filepath in enumerate(wav_files, 1):
        filename = Path(filepath).stem
        try:
            # Read audio and original sample rate via soundfile
            audio, orig_sr = sf.read(filepath)

            # Convert to mono if multi-channel
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            # Resample to target_sr (44.1 kHz) if necessary
            if orig_sr != target_sr:
                audio = resample_audio(audio, orig_sr, target_sr)

            # Check RMS energy of entire audio
            overall_rms = np.sqrt(np.mean(audio**2))
            if overall_rms < rms_threshold:
                skipped_silent += 1
                print(f"[{idx}/{total_input_files}] SKIPPED (silent, RMS={overall_rms:.5f}): {filename}")
                continue

            # If audio is shorter than target_samples, zero-pad to target_samples
            if len(audio) < target_samples:
                padded_audio = np.pad(audio, (0, target_samples - len(audio)))
                out_path = os.path.join(output_dir, f"{filename}_chunk00.wav")
                sf.write(out_path, padded_audio.astype(np.float32), target_sr)
                saved_clips += 1
            else:
                # Slicing for audio >= target_samples
                num_chunks = int(np.ceil(len(audio) / target_samples))
                for i in range(num_chunks):
                    start_idx = i * target_samples
                    end_idx = start_idx + target_samples
                    chunk = audio[start_idx:end_idx]

                    # Pad final chunk if it is shorter than target_samples
                    if len(chunk) < target_samples:
                        chunk = np.pad(chunk, (0, target_samples - len(chunk)))

                    # Check chunk RMS energy
                    chunk_rms = np.sqrt(np.mean(chunk**2))
                    if chunk_rms >= rms_threshold:
                        out_path = os.path.join(output_dir, f"{filename}_chunk{i:02d}.wav")
                        sf.write(out_path, chunk.astype(np.float32), target_sr)
                        saved_clips += 1
                    else:
                        skipped_silent += 1

            if idx % 50 == 0 or idx == total_input_files:
                print(f"Progress: [{idx}/{total_input_files}] files processed | Clips generated: {saved_clips}")

        except Exception as e:
            print(f"Error processing {filepath}: {e}")

    print("=" * 65)
    print(f"SUMMARY:")
    print(f"  Total Raw Files Processed : {total_input_files}")
    print(f"  Total 3s Clips Generated   : {saved_clips}")
    print(f"  Silent Clips Skipped      : {skipped_silent}")
    print(f"  Clips Saved To            : {os.path.abspath(output_dir)}")

    # Optionally sync/copy to data/train/percussion for model training
    if also_train:
        train_percussion_dir = "data/train/percussion"
        os.makedirs(train_percussion_dir, exist_ok=True)
        import shutil
        for f in glob.glob(os.path.join(output_dir, "*.wav")):
            shutil.copy(f, os.path.join(train_percussion_dir, os.path.basename(f)))
        print(f"  Copied {saved_clips} clips to train directory: {os.path.abspath(train_percussion_dir)}")

    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert audio files to 3s 44.1kHz clips and map to data/percussion folder."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/raw/beijing_opera_percussion_instrument_1.0",
        help="Path to raw audio folder.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/percussion",
        help="Target folder for 3s clips.",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=44100,
        help="Target sample rate in Hz (default: 44100).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Target duration in seconds (default: 3.0).",
    )
    parser.add_argument(
        "--rms_threshold",
        type=float,
        default=0.002,
        help="RMS threshold below which audio/chunks are considered silent.",
    )
    parser.add_argument(
        "--no_train_copy",
        action="store_true",
        help="Do not copy generated clips to data/train/percussion.",
    )

    args = parser.parse_args()

    process_and_slice_audio(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_sr=args.sample_rate,
        duration_sec=args.duration,
        rms_threshold=args.rms_threshold,
        also_train=not args.no_train_copy,
    )

