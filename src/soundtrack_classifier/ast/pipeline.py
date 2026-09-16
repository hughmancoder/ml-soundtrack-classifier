import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import yaml
import numpy as np
import torch

from soundtrack_classifier.ast.standardization import AudioStandardizer
from soundtrack_classifier.ast.segment_augment import SegmentAugmentor
from soundtrack_classifier.ast.feature_extraction import FrozenFeatureExtractor
from soundtrack_classifier.ast.probing_head import ProbingHeadTrainer, ProbingHead
from soundtrack_classifier.ast.separation import FilmStemSeparator
from soundtrack_classifier.ast.scoring import ContinuousFilmScorer
from soundtrack_classifier.ast.postprocessing import TemporalPostProcessor


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Loads configuration from YAML file."""
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


class OperationalPipeline:
    """
    Master 7-Phase Operational Pipeline Orchestrator.
    Config-driven end-to-end execution from raw training audio to timestamped film annotation.
    """
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.ast_cfg = self.config.get("ast", {})
        self.train_labels = self.config.get("train_labels", [
            "erhu", "pipa", "dizi", "percussion", "strings_western", "sheng",
            "suona", "guzheng", "yangqin", "woodwind", "brass", "nonmusic"
        ])
        self.paths_cfg = self.ast_cfg.get("paths", {})

    def phase_1_standardize(self, data_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """Phase 1: Audio Standardization"""
        std_cfg = self.ast_cfg.get("standardization", {})
        in_dir = data_dir or self.paths_cfg.get("data_dir", "data/train")
        out_dir = self.paths_cfg.get("standardized_dir", "data/ast/standardized")

        standardizer = AudioStandardizer(
            sr_mert=std_cfg.get("sr_mert", 24000),
            sr_clap=std_cfg.get("sr_clap", 48000),
            pre_roll_sec=std_cfg.get("pre_roll_sec", 0.05),
            post_roll_sec=std_cfg.get("post_roll_sec", 2.0),
            silence_top_db=std_cfg.get("silence_top_db", 40.0)
        )
        return standardizer.process_directory(in_dir, output_dir=out_dir)

    def phase_2_slice_and_augment(self, standardized_dir: Optional[str] = None) -> str:
        """Phase 2: Segment Slicing & Augmentation"""
        aug_cfg = self.ast_cfg.get("segment_augment", {})
        std_dir = standardized_dir or self.paths_cfg.get("standardized_dir", "data/ast/standardized")
        out_buffer = self.paths_cfg.get("segments_dir", "data/ast/augmented_segments")

        augmentor = SegmentAugmentor(
            class_labels=self.train_labels,
            window_sec=aug_cfg.get("window_sec", 4.0),
            hop_sec=aug_cfg.get("hop_sec", 1.0),
            sr_mert=self.ast_cfg.get("standardization", {}).get("sr_mert", 24000),
            sr_clap=self.ast_cfg.get("standardization", {}).get("sr_clap", 48000),
            mixup_prob=aug_cfg.get("mixup_prob", 0.5),
            mixup_alpha_range=(0.3, 0.7),
            bandpass_low_hz=aug_cfg.get("bandpass_low_hz", 80.0),
            bandpass_high_hz=aug_cfg.get("bandpass_high_hz", 7500.0),
            hum_freq_hz=aug_cfg.get("hum_frequency_hz", 50.0),
            hum_level=aug_cfg.get("hum_level", 0.005)
        )
        return augmentor.process_standardized_directory(std_dir, output_buffer_dir=out_buffer)

    def phase_3_extract_features(self, manifest_path: Optional[str] = None) -> Tuple[str, str, str]:
        """Phase 3: Frozen Feature Extraction"""
        feat_cfg = self.ast_cfg.get("feature_extraction", {})
        man_path = manifest_path or os.path.join(self.paths_cfg.get("segments_dir", "data/ast/augmented_segments"), "segments_manifest.pt")
        out_features = self.paths_cfg.get("features_dir", "data/ast/features")

        extractor = FrozenFeatureExtractor(
            mert_model_id=feat_cfg.get("mert_model_id", "m-a-p/MERT-v1-95M"),
            clap_model_id=feat_cfg.get("clap_model_id", "laion/clap-htsat-unfused"),
            device=feat_cfg.get("device", "mps"),
            fp16=feat_cfg.get("fp16", True),
            batch_size=feat_cfg.get("batch_size", 16)
        )
        return extractor.extract_from_manifest(man_path, output_features_dir=out_features)

    def phase_4_train_probing_head(
        self,
        features_dir: Optional[str] = None,
        save_checkpoint: Optional[str] = None
    ) -> ProbingHead:
        """Phase 4: Probing Head Training"""
        head_cfg = self.ast_cfg.get("probing_head", {})
        feat_dir = Path(features_dir or self.paths_cfg.get("features_dir", "data/ast/features"))
        ckpt_path = save_checkpoint or self.paths_cfg.get("model_checkpoint", "models/ast_probing_head.pt")

        x_path = feat_dir / "X.pt"
        y_path = feat_dir / "Y.pt"
        g_path = feat_dir / "groups.pt"

        if not x_path.exists() or not y_path.exists():
            raise FileNotFoundError(f"Feature matrices not found in {feat_dir}")

        X = torch.load(str(x_path)).numpy()
        Y = torch.load(str(y_path)).numpy()
        groups = torch.load(str(g_path)) if g_path.exists() else np.arange(len(X))

        trainer = ProbingHeadTrainer(
            in_dim=head_cfg.get("in_dim", 1280),
            hidden_dims=head_cfg.get("hidden_dims", [512, 256]),
            num_classes=head_cfg.get("num_classes", 12),
            dropouts=head_cfg.get("dropouts", [0.4, 0.3]),
            lr=float(head_cfg.get("lr", 0.001)),
            weight_decay=float(head_cfg.get("weight_decay", 0.01)),
            epochs=int(head_cfg.get("epochs", 40)),
            batch_size=int(head_cfg.get("batch_size", 64)),
            early_stopping_patience=int(head_cfg.get("early_stopping_patience", 10)),
            n_splits=int(head_cfg.get("n_splits", 5))
        )
        model, _ = trainer.train_kfold(X, Y, groups=groups, save_checkpoint_path=ckpt_path)
        return model

    def phase_5_separate_stems(self, film_audio_path: Optional[str] = None) -> str:
        """Phase 5: Film Stem Separation"""
        sep_cfg = self.ast_cfg.get("separation", {})
        in_path = film_audio_path or sep_cfg.get("film_audio_path", "data/test/come-drink-with-me/Come Drink With Me.wav")
        out_dir = sep_cfg.get("output_dir", "data/stems")

        separator = FilmStemSeparator(
            model_name=sep_cfg.get("model_name", "htdemucs"),
            output_dir=out_dir,
            music_stem_name=sep_cfg.get("music_stem", "other")
        )
        return separator.separate(in_path)

    def phase_6_score_film(
        self,
        music_stem_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Phase 6: Continuous Film Scoring"""
        score_cfg = self.ast_cfg.get("scoring", {})
        ckpt = checkpoint_path or self.paths_cfg.get("model_checkpoint", "models/ast_probing_head.pt")
        out_probs = self.paths_cfg.get("probabilities_path", "reports/ast_probing_raw_probabilities.npy")

        scorer = ContinuousFilmScorer(
            probing_head_checkpoint=ckpt,
            window_sec=score_cfg.get("window_sec", 4.0),
            hop_sec=score_cfg.get("hop_sec", 0.5),
            batch_size=score_cfg.get("batch_size", 32)
        )
        return scorer.score_audio_file(music_stem_path, output_probabilities_path=out_probs)

    def phase_7_postprocess(
        self,
        probabilities_path: Optional[str] = None,
        timestamps_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Phase 7: Temporal Post-Processing"""
        post_cfg = self.ast_cfg.get("postprocessing", {})
        probs_path = probabilities_path or self.paths_cfg.get("probabilities_path", "reports/ast_probing_raw_probabilities.npy")
        ts_path = timestamps_path or str(probs_path).replace(".npy", "_timestamps.npy")

        if not os.path.exists(probs_path):
            raise FileNotFoundError(f"Probabilities matrix not found: {probs_path}")

        P = np.load(probs_path)
        timestamps = np.load(ts_path) if os.path.exists(ts_path) else None

        processor = TemporalPostProcessor(
            class_labels=self.train_labels,
            p_high=float(post_cfg.get("p_high", 0.65)),
            p_low=float(post_cfg.get("p_low", 0.35)),
            median_filter_sec=float(post_cfg.get("median_filter_sec", 1.5)),
            min_duration_sec=float(post_cfg.get("min_duration_sec", 1.0)),
            hop_sec=float(self.ast_cfg.get("scoring", {}).get("hop_sec", 0.5))
        )
        events = processor.process(P, timestamps)

        json_out = self.paths_cfg.get("annotation_json", "reports/ast_inference_eval_come_drink_with_me.json")
        csv_out = self.paths_cfg.get("annotation_csv", "reports/ast_inference_eval_come_drink_with_me.csv")
        processor.export_annotations(events, json_path=json_out, csv_path=csv_out)
        return events

    def run_training_pipeline(self):
        """Runs Phases 1 through 4 (Training Pipeline)."""
        print("=== Running AST Probing Training Pipeline (Phases 1 - 4) ===")
        self.phase_1_standardize()
        self.phase_2_slice_and_augment()
        self.phase_3_extract_features()
        self.phase_4_train_probing_head()
        print("=== Training Pipeline Complete ===")

    def run_scoring_pipeline(self, film_audio_path: Optional[str] = None):
        """Runs Phases 5 through 7 (Scoring & Annotation Pipeline)."""
        print("=== Running AST Film Scoring Pipeline (Phases 5 - 7) ===")
        stem_path = self.phase_5_separate_stems(film_audio_path)
        self.phase_6_score_film(stem_path)
        self.phase_7_postprocess()
        print("=== Film Scoring & Annotation Complete ===")

    def run_full_pipeline(self, film_audio_path: Optional[str] = None):
        """Runs all 7 phases end-to-end."""
        self.run_training_pipeline()
        self.run_scoring_pipeline(film_audio_path)
