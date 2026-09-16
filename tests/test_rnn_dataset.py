import os
import pytest
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
