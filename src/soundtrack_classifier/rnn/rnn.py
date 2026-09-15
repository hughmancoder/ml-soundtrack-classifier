import torch
import torch.nn as nn
from nnAudio.features.cqt import CQT2010v2

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, freq_pool=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(freq_pool, 1)) # Preserves time resolution
        )

    def forward(self, x):
        return self.conv(x)

class ArchivalInstrumentCRNN(nn.Module):
    def __init__(self, num_classes=12, sr=16000, n_bins=168, bins_per_octave=24):
        super().__init__()
        # GPU/MPS on-the-fly CQT layer
        self.cqt = CQT2010v2(
            sr=sr,
            hop_length=512,
            fmin=65.4,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            output_format="Magnitude",
            verbose=False
        )
        
        # 2D Feature Extractor
        self.encoder = nn.Sequential(
            ConvBlock(in_ch=1, out_ch=32, freq_pool=2),   # Bins: 168 -> 84
            ConvBlock(in_ch=32, out_ch=64, freq_pool=2),  # Bins: 84 -> 42
            ConvBlock(in_ch=64, out_ch=128, freq_pool=2), # Bins: 42 -> 21
            ConvBlock(in_ch=128, out_ch=128, freq_pool=3) # Bins: 21 -> 7
        )
        
        # Frequency projection to RNN dimension
        self.fc_in = nn.Linear(128 * 7, 256)
        
        # Temporal sequence tracking
        self.rnn = nn.GRU(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Frame-level multi-label classifier
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, waveforms):
        # waveforms: (Batch, Audio_Samples)
        # Compute CQT magnitude: (B, Freq, Time)
        specs = self.cqt(waveforms)
        
        # Log scaling
        specs = torch.log1p(10.0 * specs).unsqueeze(1) # Shape: (B, 1, F, T)
        
        # CNN extraction
        feat = self.encoder(specs) # Shape: (B, 128, 7, T)
        b, c, f, t = feat.shape
        
        # Reshape for sequential processing: (B, T, C * F)
        feat = feat.permute(0, 3, 1, 2).contiguous().view(b, t, -1)
        feat = torch.relu(self.fc_in(feat))
        
        # Recurrent temporal modeling
        rnn_out, _ = self.rnn(feat) # Shape: (B, T, 256)
        
        # Frame-level logits
        logits = self.classifier(rnn_out) # Shape: (B, T, num_classes)
        return logits