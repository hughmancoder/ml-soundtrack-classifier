import os
import pytest
import soundfile as sf
import torch

from soundtrack_classifier.rnn.rnn import ArchivalInstrumentCRNN
from soundtrack_classifier.rnn.inference import run_streaming_rnn_inference


def test_streaming_rnn_inference(tmp_path):
    sr = 22050
    duration_sec = 22.0
    audio = torch.randn(int(sr * duration_sec)).numpy()
    audio_path = str(tmp_path / "test_track.wav")
    sf.write(audio_path, audio, sr)

    idx_to_class = {0: "erhu", 1: "dizi", 2: "pipa", 3: "percussion"}
    model = ArchivalInstrumentCRNN(num_classes=4, in_channels=2, sr=sr)

    events = run_streaming_rnn_inference(
        audio_path=audio_path,
        model=model,
        idx_to_class=idx_to_class,
        sr=sr,
        window_sec=10.0,
        hop_sec=2.5,
        confidence_threshold=0.0 # Capture all events for test
    )

    assert isinstance(events, list)
    for event in events:
        assert "start_sec" in event
        assert "end_sec" in event
        assert "duration_sec" in event
        assert "instrument" in event
        assert "confidence" in event
        assert event["start_sec"] >= 0.0
        assert event["end_sec"] <= duration_sec + 1.0
        assert event["instrument"] in idx_to_class.values()


def test_dynamic_config_labels_inference(tmp_path):
    sr = 22050
    audio = torch.randn(int(sr * 12.0)).numpy()
    audio_path = str(tmp_path / "test_track.wav")
    sf.write(audio_path, audio, sr)

    custom_labels = ["flute", "gong", "nonmusic"]
    model = ArchivalInstrumentCRNN(num_classes=3, in_channels=2, sr=sr)

    # 1. Verify dynamic resolution from custom config dict
    custom_cfg = {
        "train_labels": custom_labels,
        "crnn": {
            "audio": {"sample_rate": sr, "window_sec": 10.0, "hop_sec": 2.5},
            "inference": {"confidence_threshold": 0.0, "nonmusic_label": "nonmusic"}
        }
    }
    events = run_streaming_rnn_inference(
        audio_path=audio_path,
        model=model,
        idx_to_class=None,
        config=custom_cfg
    )
    assert isinstance(events, list)
    for event in events:
        assert event["instrument"] in ["flute", "gong"]

    # 2. Verify ValueError is raised when labels cannot be resolved from config or model
    empty_cfg = {"crnn": {"audio": {"sample_rate": sr}}}
    with pytest.raises(ValueError, match="No class labels found"):
        run_streaming_rnn_inference(
            audio_path=audio_path,
            model=model,
            idx_to_class=None,
            config=empty_cfg
        )

