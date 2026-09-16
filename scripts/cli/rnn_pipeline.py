import argparse
import json
import os
import sys
from pathlib import Path
import yaml
import torch

from soundtrack_classifier.rnn.rnn import ArchivalInstrumentCRNN
from soundtrack_classifier.rnn.inference import run_streaming_rnn_inference


def load_config(config_path="config/config.yaml"):
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description="Streaming 10s Window RNN Inference Pipeline")
    parser.add_argument("input", type=str, help="Path to input audio or video file")
    parser.add_argument("--model", type=str, default="models/crnn_best.pt", help="Path to trained CRNN checkpoint")
    parser.add_argument("--window-sec", type=float, default=10.0, help="Window buffer size in seconds (default: 10.0)")
    parser.add_argument("--hop-sec", type=float, default=4.0, help="Hop size in seconds (default: 4.0 for 50-60% overlap)")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size (default: 64)")
    parser.add_argument("--sr", type=int, default=22050, help="Sample rate (default: 22050)")
    parser.add_argument("--threshold", type=float, default=0.35, help="Confidence threshold (default: 0.35)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path (optional)")
    parser.add_argument("--include-nonmusic", action="store_true", help="Include nonmusic events in output")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
        sys.exit(1)

    config = load_config()
    default_labels = config.get("train_labels", [
        "sheng", "dizi", "erhu", "pipa", "suona", "guzheng",
        "yangqin", "percussion", "strings", "woodwind", "brass", "nonmusic"
    ])

    device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_name)
    print(f"Running Streaming CRNN pipeline on device: {device}")

    # Fallback to models/crnn.pt if models/crnn_best.pt not found
    model_path = args.model
    if not os.path.exists(model_path) and os.path.exists("models/crnn.pt"):
        model_path = "models/crnn.pt"

    labels = default_labels
    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "classes" in checkpoint:
            labels = checkpoint["classes"]
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        model = ArchivalInstrumentCRNN(num_classes=len(labels), sr=args.sr).to(device)
        model.load_state_dict(state_dict)
    else:
        print(f"Warning: Model weights not found at {model_path}. Initializing untrained model for pipeline verification.")
        model = ArchivalInstrumentCRNN(num_classes=len(labels), sr=args.sr).to(device)

    idx_to_class = {i: c for i, c in enumerate(labels)}

    # Run streaming inference
    events = run_streaming_rnn_inference(
        audio_path=args.input,
        model=model,
        idx_to_class=idx_to_class,
        sr=args.sr,
        window_sec=args.window_sec,
        hop_sec=args.hop_sec,
        batch_size=args.batch_size,
        device=device_name,
        confidence_threshold=args.threshold,
        include_nonmusic=args.include_nonmusic
    )

    # Output formatting
    stem = Path(args.input).stem
    out_path = args.output or f"reports/rnn_inference_{stem}.json"
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(events, f, indent=2)
    print(f"Saved inference results to {out_path}")

    # Print summary table
    print("\n" + "=" * 65)
    print(f"{'Start (s)':<12}{'End (s)':<12}{'Duration':<12}{'Instrument':<16}{'Confidence'}")
    print("-" * 65)
    for e in events[:30]:
        print(f"{e['start_sec']:<12.2f}{e['end_sec']:<12.2f}{e['duration_sec']:<12.2f}{e['instrument']:<16}{e['confidence']:.2f}")
    if len(events) > 30:
        print(f"... and {len(events) - 30} more events (see {out_path})")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
