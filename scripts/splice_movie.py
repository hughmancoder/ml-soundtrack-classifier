#!/usr/bin/env python3
"""
Splice a movie MP4 file into 15-minute MP4 clips using ffmpeg stream copying.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def splice_video(
    input_path: str = "data/raw/Come Drink With Me.mp4",
    output_dir: str = "data/raw/Come Drink With Me",
    segment_duration_sec: int = 900,  # 15 minutes = 900 seconds
):
    input_file = Path(input_path)

    if not input_file.exists():
        print(f"Error: Input video file not found at '{input_file}'")
        sys.exit(1)

    # Check if ffmpeg is available
    if not shutil.which("ffmpeg"):
        print(
            "Error: 'ffmpeg' command not found. Please install ffmpeg to use this script."
        )
        sys.exit(1)

    # Ensure output directory exists
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_name = input_file.stem  # e.g., "Come Drink With Me"
    output_pattern = out_dir / f"{base_name}_part_%02d.mp4"

    print(f"Input file: {input_file}")
    print(f"Output directory: {out_dir}")
    print(f"Segment duration: {segment_duration_sec} seconds ({segment_duration_sec // 60} minutes)")
    print("Splicing video using ffmpeg stream copy...")

    # ffmpeg command using segment muxer with stream copying for fast, lossless splitting
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_file),
        "-c",
        "copy",
        "-map",
        "0",
        "-segment_time",
        str(segment_duration_sec),
        "-f",
        "segment",
        "-reset_timestamps",
        "1",
        str(output_pattern),
    ]

    try:
        subprocess.run(cmd, check=True)
        print("\nSplicing completed successfully!")

        # List generated files
        generated_files = sorted(list(out_dir.glob("*.mp4")))
        print(f"\nGenerated {len(generated_files)} file(s) in '{out_dir}':")
        for f in generated_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  - {f.name} ({size_mb:.2f} MB)")

    except subprocess.CalledProcessError as e:
        print(f"Error during video splicing: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Splice video into 15-minute MP4 chunks."
    )
    parser.add_argument(
        "--input",
        "-i",
        default="data/raw/Come Drink With Me.mp4",
        help="Path to input video file (default: data/raw/Come Drink With Me.mp4)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data/raw/Come Drink With Me",
        help="Path to output directory (default: data/raw/Come Drink With Me)",
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=int,
        default=900,
        help="Segment duration in seconds (default: 900 for 15 minutes)",
    )

    args = parser.parse_args()
    splice_video(
        input_path=args.input,
        output_dir=args.output,
        segment_duration_sec=args.duration,
    )


if __name__ == "__main__":
    main()
