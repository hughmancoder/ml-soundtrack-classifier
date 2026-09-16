import os
import pytest
import numpy as np
import soundfile as sf
import torch
from soundtrack_classifier.rnn.dataset import ArchivalSoundtrackRNNDataset


def test_rnn_dataset_dynamic_crop(tmp_path):
    sr = 22050
    crop_duration_sec = 8.0
    expected_samples = int(sr * crop_duration_sec)

    # Create dummy classes and wav files
    class_names = ["erhu", "dizi"]
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    for c in class_names:
        c_dir = tmp_path / c
        c_dir.mkdir()
        # Create a 12s file (longer than crop)
        audio_long = torch.randn(int(sr * 12.0)).numpy()
        sf.write(str(c_dir / "long.wav"), audio_long, sr)

        # Create a 3s file (shorter than crop, should be padded)
        audio_short = torch.randn(int(sr * 3.0)).numpy()
        sf.write(str(c_dir / "short.wav"), audio_short, sr)

    dataset = ArchivalSoundtrackRNNDataset(
        data_dir=str(tmp_path),
        class_to_idx=class_to_idx,
        sr=sr,
        crop_duration_sec=crop_duration_sec,
        random_crop=True
    )

    assert len(dataset) == 4

    for i in range(len(dataset)):
        waveform, label = dataset[i]
        assert waveform.shape[0] == expected_samples
        assert label in [0, 1]


def test_pad_short_audio_random_offset():
    sr = 22050
    crop_duration_sec = 8.0
    num_samples = int(sr * crop_duration_sec)

    dataset = ArchivalSoundtrackRNNDataset(
        data_dir="data/train",
        class_to_idx={"erhu": 0},
        sr=sr,
        crop_duration_sec=crop_duration_sec,
        random_crop=True
    )

    # 1.5 second short event (e.g. gong strike)
    event_len = int(1.5 * sr)
    short_event = torch.ones(event_len).numpy()

    padded = dataset._pad_short_audio(short_event)

    assert padded.shape[0] == num_samples
    # Verify that the short event is preserved without tiling (event length matches and surrounds are optical noise floor)
    # High amplitude region corresponds to event
    high_amp_mask = np.abs(padded) > 0.5
    assert np.isclose(np.sum(high_amp_mask), event_len, atol=2)
    # Background region has low-level optical noise floor (< 0.05)
    bg_mask = ~high_amp_mask
    assert np.all(np.abs(padded[bg_mask]) < 0.05)
    # Ensure background is not absolute zero
    assert not np.all(padded[bg_mask] == 0.0)

