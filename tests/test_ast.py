import os
import tempfile
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
import torch

from soundtrack_classifier.ast.cqt_transformer import PositionalEncoding, CQTTransformer
from soundtrack_classifier.ast.probing_head import ProbingHead, ProbingHeadTrainer
from soundtrack_classifier.ast.standardization import AudioStandardizer
from soundtrack_classifier.ast.segment_augment import SegmentAugmentor, OpticalChannelFilter
from soundtrack_classifier.ast.feature_extraction import FrozenFeatureExtractor
from soundtrack_classifier.ast.postprocessing import TemporalPostProcessor
from soundtrack_classifier.ast.pipeline import OperationalPipeline, load_config


def test_positional_encoding_and_cqt_transformer():
    d_model = 256
    pos = PositionalEncoding(d_model=d_model, max_len=100)
    x = torch.zeros(50, 4, d_model)  # (seq_len, batch_size, d_model)
    out = pos(x)
    assert out.shape == (50, 4, d_model)

    model = CQTTransformer(
        n_classes=12,
        n_channels=2,
        n_bins=168,
        d_model=d_model,
        n_heads=4,
        n_layers=2
    )
    dummy_cqt = torch.randn(2, 2, 168, 60)  # (Batch, Channels, Bins, Time)
    logits = model(dummy_cqt)
    assert logits.shape == (2, 12)


def test_probing_head_forward():
    head = ProbingHead(
        in_dim=1280,
        hidden_dims=[512, 256],
        num_classes=12,
        dropouts=[0.4, 0.3]
    )
    x = torch.randn(8, 1280)
    logits = head(x)
    assert logits.shape == (8, 12)

    probs = head.predict_proba(x)
    assert probs.shape == (8, 12)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_probing_head_trainer():
    trainer = ProbingHeadTrainer(
        in_dim=1280,
        hidden_dims=[64, 32],
        num_classes=12,
        epochs=3,
        batch_size=16,
        n_splits=2,
        device="cpu"
    )
    np.random.seed(42)
    X = np.random.randn(40, 1280).astype(np.float32)
    Y = (np.random.rand(40, 12) > 0.8).astype(np.float32)
    groups = np.repeat(np.arange(10), 4)

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, "test_probing.pt")
        model, metrics = trainer.train_kfold(X, Y, groups=groups, save_checkpoint_path=ckpt_path)
        assert os.path.exists(ckpt_path)
        assert len(metrics) == 2
        for m in metrics:
            assert "val_loss" in m
            assert "f1_macro" in m


def test_audio_standardizer(tmp_path):
    standardizer = AudioStandardizer(
        sr_mert=24000,
        sr_clap=48000,
        pre_roll_sec=0.05,
        post_roll_sec=0.1,
        silence_top_db=20.0
    )

    # Create synthetic 1s audio with leading/trailing silence
    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    signal_wave = np.sin(2 * np.pi * 440 * t)
    # Add silence before and after
    audio = np.concatenate([np.zeros(sr // 2), signal_wave, np.zeros(sr // 2)])

    in_file = tmp_path / "test_raw.wav"
    sf.write(str(in_file), audio.astype(np.float32), sr)

    out_24k = tmp_path / "out_24k"
    out_48k = tmp_path / "out_48k"

    result = standardizer.standardize_file(
        input_path=str(in_file),
        output_dir_24k=str(out_24k),
        output_dir_48k=str(out_48k),
        class_name="erhu"
    )

    assert os.path.exists(result["path_24k"])
    assert os.path.exists(result["path_48k"])

    assert sf.info(result["path_24k"]).subtype == "FLOAT"
    assert sf.info(result["path_48k"]).subtype == "FLOAT"

    data_24k, sr_24 = sf.read(result["path_24k"], dtype="float32")
    assert sr_24 == 24000
    assert data_24k.dtype == np.float32

    data_48k, sr_48 = sf.read(result["path_48k"], dtype="float32")
    assert sr_48 == 48000
    assert data_48k.dtype == np.float32


def test_segment_augmentor(tmp_path):
    augmentor = SegmentAugmentor(
        class_labels=["erhu", "pipa", "dizi"],
        window_sec=4.0,
        hop_sec=1.0,
        sr_mert=24000,
        sr_clap=48000,
        mixup_prob=1.0
    )

    # Generate 6 seconds of audio
    len_24k = int(6.0 * 24000)
    len_48k = int(6.0 * 48000)
    audio_24k = np.sin(np.linspace(0, 100, len_24k)).astype(np.float32)
    audio_48k = np.sin(np.linspace(0, 100, len_48k)).astype(np.float32)

    p24 = tmp_path / "erhu_24k.wav"
    p48 = tmp_path / "erhu_48k.wav"
    sf.write(str(p24), audio_24k, 24000)
    sf.write(str(p48), audio_48k, 48000)

    segments = augmentor.slice_file(
        path_24k=str(p24),
        path_48k=str(p48),
        class_name="erhu",
        recording_id="rec_01"
    )
    assert len(segments) >= 2
    for s in segments:
        assert len(s["audio_24k"]) == 4 * 24000
        assert len(s["audio_48k"]) == 4 * 48000
        assert s["labels"][0] == 1.0  # erhu is idx 0

    augmented = augmentor.augment_segments(segments)
    assert len(augmented) > len(segments)


def test_optical_channel_filter():
    filt = OpticalChannelFilter(sr=24000, low_hz=80.0, high_hz=7500.0, hum_freq_hz=50.0, hum_level=0.01)
    x = np.random.randn(24000).astype(np.float32)
    filtered = filt.apply_filter(x)
    assert filtered.shape == x.shape
    with_hum = filt.inject_hum(filtered)
    assert with_hum.shape == x.shape


def test_temporal_postprocessor(tmp_path):
    labels = ["erhu", "pipa"]
    processor = TemporalPostProcessor(
        class_labels=labels,
        p_high=0.65,
        p_low=0.35,
        median_filter_sec=1.5,
        min_duration_sec=1.0,
        hop_sec=0.5
    )

    # 10 seconds of probabilities (20 frames at 0.5s hop)
    T = 20
    P = np.zeros((T, 2), dtype=np.float32)
    # Erhu active from frame 4 to 12 (2.0s to 6.0s = 4.0s duration) with high confidence
    P[4:12, 0] = 0.85
    # Pipa active for only 1 frame (0.5s duration) -> should be pruned
    P[16, 1] = 0.90

    events = processor.process(P)
    # Only erhu should survive the min_duration filter
    assert len(events) == 1
    assert events[0]["instrument"] == "erhu"
    assert events[0]["start_time"] >= 1.5
    assert events[0]["duration"] >= 1.0

    json_path = tmp_path / "out.json"
    csv_path = tmp_path / "out.csv"
    processor.export_annotations(events, json_path=str(json_path), csv_path=str(csv_path))
    assert json_path.exists()
    assert csv_path.exists()


def test_config_ast_section():
    config = load_config("config/config.yaml")
    assert "ast" in config
    ast_cfg = config["ast"]
    assert "standardization" in ast_cfg
    assert "segment_augment" in ast_cfg
    assert "feature_extraction" in ast_cfg
    assert "probing_head" in ast_cfg
    assert "separation" in ast_cfg
    assert "scoring" in ast_cfg
    assert "postprocessing" in ast_cfg
    assert "paths" in ast_cfg


def test_sharded_segment_processing(tmp_path):
    # Create mock standardized structure
    std_dir = tmp_path / "standardized"
    dir_24 = std_dir / "24k" / "erhu"
    dir_48 = std_dir / "48k" / "erhu"
    dir_24.mkdir(parents=True)
    dir_48.mkdir(parents=True)

    sr_24, sr_48 = 24000, 48000
    # 6 seconds of audio
    wav_24 = np.sin(np.linspace(0, 50, 6 * sr_24)).astype(np.float32)
    wav_48 = np.sin(np.linspace(0, 50, 6 * sr_48)).astype(np.float32)
    sf.write(str(dir_24 / "test.wav"), wav_24, sr_24)
    sf.write(str(dir_48 / "test.wav"), wav_48, sr_48)

    augmentor = SegmentAugmentor(
        class_labels=["erhu", "pipa"],
        window_sec=4.0,
        hop_sec=1.0,
        mixup_prob=0.5
    )

    out_buf = tmp_path / "sharded_segments"
    manifest_path = augmentor.process_standardized_directory(
        standardized_dir=str(std_dir),
        output_buffer_dir=str(out_buf),
        shard_size=2  # small shard size to test sharding
    )

    assert os.path.exists(manifest_path)
    manifest = torch.load(manifest_path, weights_only=False)
    assert "shards" in manifest
    assert len(manifest["shards"]) >= 2
    for s_path in manifest["shards"]:
        assert os.path.exists(s_path)
        sh_data = torch.load(s_path, weights_only=False)
        assert len(sh_data) <= 2
        for seg in sh_data:
            assert "audio_24k" in seg
            assert "audio_48k" in seg
            assert seg["audio_24k"].shape == (4 * sr_24,)

