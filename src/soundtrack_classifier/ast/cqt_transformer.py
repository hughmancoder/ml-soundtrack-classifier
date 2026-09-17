import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Injects temporal order information into the sequence."""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape expects: (seq_len, batch_size, d_model)
        return x + self.pe[:x.size(0)]


class CQTTransformer(nn.Module):
    """
    Audio Spectrogram Transformer specifically for CQT/HPSS features.
    """
    def __init__(
        self,
        n_classes: int = 12,
        n_channels: int = 2,  # 2 for HPSS stack, 1 for standard CQT
        n_bins: int = 168,    # Matches your 7-octave, 24-bin configuration
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        # 1. Feature Projection
        # Flattens channels and frequency bins: n_channels * n_bins
        self.input_dim = n_channels * n_bins
        self.feature_projection = nn.Linear(self.input_dim, d_model)

        # 2. Positional Encoding & Transformer Encoder
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_layers)

        # 3. Multi-Label Classification Head
        self.layer_norm = nn.LayerNorm(d_model)
        self.mlp_head = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, Channels, Bins, Time) from cqt.py
        Returns:
            logits: Tensor of shape (Batch, n_classes)
        """
        B, C, F, T = x.shape

        # Reshape to (Batch, Time, Channels * Bins)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(B, T, C * F)

        # Project to the Transformer's hidden dimension
        x = self.feature_projection(x)

        # Add positional encoding (requires (Time, Batch, d_model) temporarily)
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)

        # Pass through the attention blocks
        x = self.transformer_encoder(x)

        # Temporal mean pooling (collapse the Time dimension)
        x = x.mean(dim=1)

        # Multi-label classification projection
        x = self.layer_norm(x)
        logits = self.mlp_head(x)

        return logits
