import pytest
import torch
from soundtrack_classifier.rnn.rnn import ArchivalInstrumentCRNN


def test_crnn_waveform_forward():
    num_classes = 12
    sr = 22050
    duration_sec = 8.0
    model = ArchivalInstrumentCRNN(num_classes=num_classes, in_channels=2, sr=sr)
    model.eval()

    batch_size = 2
    samples = int(sr * duration_sec)
    audio = torch.randn(batch_size, samples)

    # Test clip-level logits
    clip_logits = model(audio, return_frames=False)
    assert clip_logits.shape == (batch_size, num_classes)

    # Test frame-level logits
    frame_logits = model(audio, return_frames=True)
    assert frame_logits.dim() == 3
    assert frame_logits.shape[0] == batch_size
    assert frame_logits.shape[2] == num_classes
    assert frame_logits.shape[1] > 0


def test_crnn_features_forward_and_backward():
    num_classes = 8
    model = ArchivalInstrumentCRNN(num_classes=num_classes, in_channels=2)
    model.train()

    batch_size = 2
    n_bins = 168
    time_frames = 100
    features = torch.randn(batch_size, 2, n_bins, time_frames, requires_grad=True)
    labels = torch.tensor([1, 4])

    logits = model(features, return_frames=False)
    assert logits.shape == (batch_size, num_classes)

    loss_fn = torch.nn.BCEWithLogitsLoss()
    targets = torch.zeros((labels.size(0), num_classes), dtype=torch.float32).scatter_(1, labels.unsqueeze(1), 1.0)
    loss = loss_fn(logits, targets)
    loss.backward()

    # Check encoder, GRU, and classifier have gradients
    assert model.encoder[0].conv[0].weight.grad is not None
    assert model.rnn.weight_ih_l0.grad is not None
    assert model.classifier.weight.grad is not None


def test_binary_focal_loss_backward():
    from train_rnn import BinaryFocalLossWithLogits

    batch_size = 4
    num_classes = 6
    logits = torch.randn(batch_size, num_classes, requires_grad=True)
    labels = torch.tensor([0, 2, 5, 1])
    targets = torch.zeros((batch_size, num_classes)).scatter_(1, labels.unsqueeze(1), 1.0)
    pos_weight = torch.tensor([2.0, 1.0, 3.0, 1.0, 1.0, 2.0])

    criterion = BinaryFocalLossWithLogits(gamma=2.0, alpha=0.25, pos_weight=pos_weight)
    loss = criterion(logits, targets)

    assert loss.dim() == 0
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

    loss.backward()
    assert logits.grad is not None
    assert torch.all(torch.isfinite(logits.grad))

