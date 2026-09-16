import argparse
import os
import sys
from pathlib import Path
import json

from soundtrack_classifier.ast.pipeline import OperationalPipeline


def main():
    parser = argparse.ArgumentParser(description="End-to-End AST & Foundation Model Pipeline CLI")
    parser.add_argument("input", nargs="?", default=None, help="Path to input audio file or film for scoring (optional for training phases)")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--phase",
        type=str,
        default="all",
        choices=["1", "2", "3", "4", "5", "6", "7", "train", "score", "all"],
        help="Pipeline phase to execute: 1 (Standardize), 2 (Augment), 3 (Features), 4 (Train), 5 (Separate), 6 (Score), 7 (Postprocess), train (1-4), score (5-7), all (1-7)"
    )
    parser.add_argument("--model", type=str, default=None, help="Path to probing head checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Path to output JSON annotations")
    parser.add_argument("--csv", type=str, default=None, help="Path to output CSV annotations")

    args = parser.parse_args()

    pipeline = OperationalPipeline(config_path=args.config)

    # Resolve input film audio path
    input_path = args.input
    if not input_path and args.phase in ("5", "6", "score", "all"):
        input_path = pipeline.ast_cfg.get("separation", {}).get("film_audio_path")
        if not input_path or not os.path.exists(input_path):
            input_path = "data/test/come-drink-with-me/Come Drink With Me.wav"

    print(f"Executing AST Pipeline [Phase: {args.phase}]...")

    if args.phase == "1":
        pipeline.phase_1_standardize()
    elif args.phase == "2":
        pipeline.phase_2_slice_and_augment()
    elif args.phase == "3":
        pipeline.phase_3_extract_features()
    elif args.phase == "4":
        pipeline.phase_4_train_probing_head(save_checkpoint=args.model)
    elif args.phase == "5":
        if not input_path or not os.path.exists(input_path):
            print(f"Error: Input audio not found at {input_path}")
            sys.exit(1)
        pipeline.phase_5_separate_stems(input_path)
    elif args.phase == "6":
        stem_path = input_path
        if input_path and not ("music" in input_path or "other" in input_path):
            # Check if separated stem exists
            stem_path = pipeline.phase_5_separate_stems(input_path)
        pipeline.phase_6_score_film(stem_path, checkpoint_path=args.model)
    elif args.phase == "7":
        events = pipeline.phase_7_postprocess()
        _print_events_table(events)
    elif args.phase == "train":
        pipeline.run_training_pipeline()
    elif args.phase == "score":
        if not input_path or not os.path.exists(input_path):
            print(f"Error: Film audio path required for scoring pipeline: {input_path}")
            sys.exit(1)
        pipeline.run_scoring_pipeline(input_path)
    elif args.phase == "all":
        pipeline.run_full_pipeline(input_path)


def _print_events_table(events):
    print("\n" + "=" * 70)
    print(f"{'Start (s)':<12}{'End (s)':<12}{'Duration':<12}{'Instrument':<18}{'Confidence'}")
    print("-" * 70)
    for e in events[:40]:
        print(f"{e['start_time']:<12.2f}{e['end_time']:<12.2f}{e['duration']:<12.2f}{e['instrument']:<18}{e['confidence']:.4f}")
    if len(events) > 40:
        print(f"... [{len(events) - 40} more events]")
    print("=" * 70)


if __name__ == "__main__":
    main()
