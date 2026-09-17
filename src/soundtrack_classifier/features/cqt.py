import os

import librosa
import torch
from nnAudio.features.cqt import CQT1992v2, CQT2010v2


def compute_hpss_from_spec(
    spec: torch.Tensor,
    kernel_time: int = 31,
    kernel_freq: int = 31,
    power: float = 2.0
) -> torch.Tensor:
    """
    Computes Harmonic-Percussive Source Separation (HPSS) masks directly on a 2D/3D spectrogram tensor.
    Uses median filtering across time to isolate sustained harmonic events (erhu, dizi, sheng)
    and across frequency to isolate broadband percussive transients (opera gongs, foley impacts, pipa strikes).

    Args:
        spec: (B, F, T) or (F, T) magnitude spectrogram tensor.
        kernel_time: Size of median filter across time axis (frames). Must be odd.
        kernel_freq: Size of median filter across frequency axis (bins). Must be odd.
        power: Exponent for soft masking (default: 2.0).

    Returns:
        (B, 2, F, T) or (2, F, T) tensor where:
            Channel 0: Harmonic component
            Channel 1: Percussive component
    """
    was_2d = False
    if spec.dim() == 2:
        was_2d = True
        spec = spec.unsqueeze(0)

    orig_device = spec.device
    # PyTorch MPS does not have an optimized kernel for strided 4D unfold median,
    # causing severe execution latency (~20x slower than CPU).
    # Performing the 1D median filters on CPU avoids MPS kernel thrashing.
    if orig_device.type == "mps":
        spec = spec.cpu()

    B, F, T = spec.shape
    kt = min(kernel_time, T if T % 2 != 0 else max(1, T - 1))
    kf = min(kernel_freq, F if F % 2 != 0 else max(1, F - 1))
    if kt < 3:
        kt = 3 if T >= 3 else 1
    if kf < 3:
        kf = 3 if F >= 3 else 1

    if kt > 1:
        pad_t = kt // 2
        spec_pad_t = torch.nn.functional.pad(spec, (pad_t, pad_t), mode="replicate")
        windows_t = spec_pad_t.unfold(dimension=-1, size=kt, step=1)
        H_raw = windows_t.median(dim=-1).values
    else:
        H_raw = spec

    if kf > 1:
        pad_f = kf // 2
        spec_pad_f = torch.nn.functional.pad(spec.unsqueeze(1), (0, 0, pad_f, pad_f), mode="replicate").squeeze(1)
        windows_f = spec_pad_f.unfold(dimension=1, size=kf, step=1)
        P_raw = windows_f.median(dim=-1).values
    else:
        P_raw = spec

    H_pow = H_raw ** power
    P_pow = P_raw ** power
    total = H_pow + P_pow + 1e-8
    mask_H = H_pow / total
    mask_P = P_pow / total

    out = torch.stack([spec * mask_H, spec * mask_P], dim=1)
    if orig_device.type == "mps":
        out = out.to(orig_device)
    return out.squeeze(0) if was_2d else out


class CQTExtractor:
    """
    Extracts Constant-Q Transform (CQT) spectrograms from audio files and tensors.
    Configured by default for archival soundtrack classification:
    - 24 bins per octave spanning 7 octaves (168 bins total)
    - fmin ~ 65.4 Hz (C2) up to fmax ~ 8.1 kHz (covering optical film bandwidth)
    - Optional 2-channel Harmonic-Percussive Source Separation (HPSS) stack.
    """
    def __init__(
        self,
        sr: int = 22050,
        fmin: float = 65.4,
        n_bins: int = 168,
        bins_per_octave: int = 24,
        hop_length: int = 512,
        device: str = "cpu"
    ):
        self.sr = sr
        self.device = device
        self.hop_length = hop_length
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.fmin = fmin

        # Initialize nnAudio CQT2010v2 (or fallback to CQT1992v2)
        try:
            self.cqt_layer = CQT2010v2(
                sr=sr,
                hop_length=hop_length,
                fmin=fmin,
                n_bins=n_bins,
                bins_per_octave=bins_per_octave,
                output_format="Magnitude",
                verbose=False
            ).to(self.device)
        except Exception:
            self.cqt_layer = CQT1992v2(
                sr=sr,
                hop_length=hop_length,
                fmin=fmin,
                n_bins=n_bins,
                bins_per_octave=bins_per_octave,
                output_format="Magnitude",
                verbose=False
            ).to(self.device)

    def extract(self, audio_tensor: torch.Tensor, return_hpss: bool = True) -> torch.Tensor:
        """
        Extracts CQT features from an audio tensor.
        Args:
            audio_tensor: (B, samples) or (samples,)
            return_hpss: If True, returns (B, 2, n_bins, time_steps) Harmonic-Percussive stack.
                         If False, returns (B, 1, n_bins, time_steps).
        """
        was_1d = False
        if audio_tensor.dim() == 1:
            was_1d = True
            audio_tensor = audio_tensor.unsqueeze(0)

        audio_tensor = audio_tensor.to(self.device)
        spec = self.cqt_layer(audio_tensor) # (B, F, T)

        if return_hpss:
            hpss_spec = compute_hpss_from_spec(spec) # (B, 2, F, T)
            return hpss_spec.squeeze(0) if was_1d else hpss_spec
        else:
            single = spec.unsqueeze(1)
            return single.squeeze(0) if was_1d else single

    def extract_from_file(self, audio_path: str, return_hpss: bool = True) -> torch.Tensor:
        """
        Loads an audio file and computes its CQT spectrogram (optionally HPSS 2-channel).
        Returns:
            PyTorch tensor of shape (2, n_bins, time_steps) or (1, n_bins, time_steps).
        """
        audio, _ = librosa.load(audio_path, sr=self.sr, mono=True)
        audio_tensor = torch.tensor(audio, dtype=torch.float32)
        return self.extract(audio_tensor, return_hpss=return_hpss)

    def save_features(self, cqt_tensor: torch.Tensor, output_path: str):
        """
        Saves the CQT tensor to disk as a .pt file.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(cqt_tensor.cpu(), output_path)
        print(f"Saved CQT features to {output_path}")

    def load_features(self, input_path: str) -> torch.Tensor:
        """
        Loads saved CQT features from disk.
        """
        return torch.load(input_path, map_location=self.device)
