import torch
import torch.nn as nn
from nnAudio.features.cqt import CQT2010v2, CQT1992v2
from soundtrack_classifier.features.cqt import compute_hpss_from_spec


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, freq_pool: int = 2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(freq_pool, 1))  # Preserves time resolution
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ArchivalInstrumentCRNN(nn.Module):
    """
    Convolutional Recurrent Neural Network (CRNN) tailored for archival optical soundtracks.
    
    Key Features:
    - 2-Channel Front-End:
        Channel 0 (Harmonic): continuous pitched instruments (erhu, dizi, sheng, strings).
        Channel 1 (Percussive): sharp transients (pipa, yangqin, Beijing opera percussion, foley).
    - Preserved Time Resolution: MaxPool2d pools only along the frequency axis:
        168 -> 84 -> 42 -> 21 -> 7 bins.
    - Bidirectional GRU: Models sequential phrasing, sustained decays, and separates steady-state
        optical surface noise baselines from transient musical events.
    - Flexible Input: Accepts either raw waveforms (B, Samples) or pre-extracted CQT maps (B, 2, F, T).
    """
    def __init__(
        self,
        num_classes: int = 12,
        in_channels: int = 2,
        sr: int = 22050,
        n_bins: int = 168,
        bins_per_octave: int = 24,
        hop_length: int = 512,
        hidden_size: int = 128,
        num_rnn_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.2
    ):
        super().__init__()
        self.sr = sr
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.hop_length = hop_length

        # On-the-fly CQT layer for raw waveform inputs
        try:
            self.cqt = CQT2010v2(
                sr=sr,
                hop_length=hop_length,
                fmin=65.4,
                n_bins=n_bins,
                bins_per_octave=bins_per_octave,
                output_format="Magnitude",
                verbose=False
            )
        except Exception:
            self.cqt = CQT1992v2(
                sr=sr,
                hop_length=hop_length,
                fmin=65.4,
                n_bins=n_bins,
                bins_per_octave=bins_per_octave,
                output_format="Magnitude",
                verbose=False
            )

        # 2D Feature Extractor (pools frequency: 168 -> 84 -> 42 -> 21 -> 7)
        self.encoder = nn.Sequential(
            ConvBlock(in_ch=in_channels, out_ch=32, freq_pool=2),   # 168 -> 84
            ConvBlock(in_ch=32, out_ch=64, freq_pool=2),           # 84 -> 42
            ConvBlock(in_ch=64, out_ch=128, freq_pool=2),          # 42 -> 21
            ConvBlock(in_ch=128, out_ch=128, freq_pool=3)          # 21 -> 7
        )

        # Projection from flattened frequency channels to RNN input dim
        rnn_input_dim = hidden_size * 2
        self.fc_in = nn.Linear(128 * 7, rnn_input_dim)
        self.dropout = nn.Dropout(dropout)

        # Temporal sequence tracking
        self.rnn = nn.GRU(
            input_size=rnn_input_dim,
            hidden_size=hidden_size,
            num_layers=num_rnn_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_rnn_layers > 1 else 0.0
        )

        # Output projection
        rnn_out_dim = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Linear(rnn_out_dim, num_classes)

    def extract_features(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Extracts on-the-fly 2-channel log-magnitude HPSS CQT features from waveforms.
        Args:
            waveforms: (Batch, Audio_Samples)
        Returns:
            (Batch, 2, n_bins, time_frames)
        """
        specs = self.cqt(waveforms) # (B, F, T)
        
        if self.in_channels == 2:
            hpss_specs = compute_hpss_from_spec(specs) # (B, 2, F, T)
        else:
            hpss_specs = specs.unsqueeze(1) # (B, 1, F, T)
            
        # Logarithmic compression
        features = torch.log1p(10.0 * hpss_specs)
        
        # Feature Standardisation (Instance z-score per channel)
        f_mean = features.mean(dim=(-2, -1), keepdim=True)
        f_std = features.std(dim=(-2, -1), keepdim=True).clamp(min=1e-5)
        features = (features - f_mean) / f_std
        
        return features

    def forward(self, x: torch.Tensor, return_frames: bool = False) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Either raw waveform tensor (B, Samples) or 2D feature map (B, C, F, T).
            return_frames: If True, returns frame-level logits (B, T, num_classes).
                           If False, returns temporal average-pooled clip logits (B, num_classes).
        """
        if x.dim() == 2:
            # Raw audio waveform (B, Samples)
            feat_map = self.extract_features(x)
        elif x.dim() == 3:
            # (B, F, T) -> add channel dimension or split HPSS
            if self.in_channels == 2:
                feat_map = compute_hpss_from_spec(x)
            else:
                feat_map = x.unsqueeze(1)
            feat_map = torch.log1p(10.0 * feat_map)
        elif x.dim() == 4:
            # Already (B, C, F, T)
            feat_map = x
        else:
            raise ValueError(f"Unsupported input dimension {x.dim()}: expected 2D waveform or 4D feature map.")

        # CNN feature extraction
        conv_out = self.encoder(feat_map) # (B, 128, 7, T)
        b, c, f, t = conv_out.shape

        # Reshape for sequence processing: (B, T, C * F)
        seq_in = conv_out.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        seq_in = self.dropout(torch.relu(self.fc_in(seq_in)))

        # Recurrent temporal sequence modeling
        rnn_out, _ = self.rnn(seq_in) # (B, T, rnn_out_dim)

        # Frame-level predictions
        frame_logits = self.classifier(rnn_out) # (B, T, num_classes)

        if return_frames:
            return frame_logits
        else:
            # Temporal pooling over the sequence for clip-level prediction
            return torch.mean(frame_logits, dim=1)