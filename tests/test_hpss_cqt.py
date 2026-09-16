import pytest
import torch
from soundtrack_classifier.features.cqt import CQTExtractor, compute_hpss_from_spec


def test_compute_hpss_from_spec():
    # Shape: (Batch=2, Freq=168, Time=100)
    spec = torch.rand(2, 168, 100) + 0.01
    hpss = compute_hpss_from_spec(spec, kernel_time=15, kernel_freq=15)
    
    # Assert output shape is (Batch, 2, Freq, Time)
    assert hpss.shape == (2, 2, 168, 100)
    assert not torch.isnan(hpss).any()
    assert not torch.isinf(hpss).any()

    # Harmonic channel (0) and Percussive channel (1) should both be non-negative
    harmonic = hpss[:, 0, :, :]
    percussive = hpss[:, 1, :, :]
    assert (harmonic >= 0).all()
    assert (percussive >= 0).all()

    # 2D spec test
    spec_2d = torch.rand(168, 50) + 0.01
    hpss_2d = compute_hpss_from_spec(spec_2d)
    assert hpss_2d.shape == (2, 168, 50)


def test_cqt_extractor():
    sr = 22050
    duration_sec = 6.0
    extractor = CQTExtractor(sr=sr, fmin=65.4, n_bins=168, bins_per_octave=24)

    # Test single waveform extraction
    audio = torch.randn(int(sr * duration_sec))
    features = extractor.extract(audio, return_hpss=True)
    assert features.dim() == 3 # (2, n_bins, T)
    assert features.shape[0] == 2
    assert features.shape[1] == 168

    # Test batch extraction
    audio_batch = torch.randn(2, int(sr * duration_sec))
    features_batch = extractor.extract(audio_batch, return_hpss=True)
    assert features_batch.dim() == 4 # (B, 2, n_bins, T)
    assert features_batch.shape[:3] == (2, 2, 168)
