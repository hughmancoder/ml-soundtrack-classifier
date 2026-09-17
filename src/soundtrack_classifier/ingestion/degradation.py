import os
import random
from typing import Any, Dict, Optional

import numpy as np
import scipy.signal as signal

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import pedalboard
except ImportError:
    pedalboard = None


class SyntheticDegradation:
    """
    Applies subtle synthetic archival audio degradation mimicking 1960s Shaw Brothers optical film soundtracks.

    Optical Sound Channel Simulation:
    - Low-pass filter (default: 5.5 kHz Butterworth) bounded by optical slit physics.
    - High-pass filter (default: 80-100 Hz) removing subsonic rumbles.
    - Mild non-linear tanh saturation simulating 35mm optical emulsion print compression.
    - Subtle optical film surface noise & projector mains hum (50/60 Hz) mixed at gentle SNR (default: 32-42 dB).
    - Optional soundstage reverberation simulating reflective 1960s Hong Kong studios.
    """

    def __init__(self, sr: int = 22050, config: Optional[Dict[str, Any]] = None, **kwargs):
        self.sr = int(sr)
        cfg = dict(config or {})
        cfg.update(kwargs)

        # 1. Optical Channel Filter Parameters
        self.lowpass_cutoff_hz = float(cfg.get("lowpass_cutoff_hz", 5500.0))
        self.lowpass_order = int(cfg.get("lowpass_order", 2))
        self.highpass_cutoff_hz = float(cfg.get("highpass_cutoff_hz", 80.0))
        self.highpass_order = int(cfg.get("highpass_order", 1))

        # 2. Noise & Projector Hum Parameters (Subtle, gentle ambient baseline)
        self.snr_min_db = float(cfg.get("snr_min_db", 42.0))
        self.snr_max_db = float(cfg.get("snr_max_db", 52.0))
        self.hum_frequency_hz = float(cfg.get("hum_frequency_hz", 50.0))
        self.hum_level = float(cfg.get("hum_level", 0.005))
        self.crackle_probability = float(cfg.get("crackle_probability", 0.0))
        self.noise_file = cfg.get("noise_file", None)

        # 3. Non-linear Tanh Saturation (Optical Emulsion Print Compression)
        self.saturation_drive = float(cfg.get("saturation_drive", 1.03))

        # 4. Soundstage Spatial Reverb
        self.reverb_room_size = float(cfg.get("reverb_room_size", 0.15))
        self.reverb_wet_level = float(cfg.get("reverb_wet_level", 0.03))

        # Build Butterworth Filters (using Second-Order Sections for numerical stability)
        nyquist = 0.5 * self.sr
        safe_lp = min(self.lowpass_cutoff_hz, nyquist * 0.96)
        self.sos_lp = signal.butter(
            N=max(1, self.lowpass_order),
            Wn=safe_lp,
            btype="low",
            fs=self.sr,
            output="sos"
        )

        if self.highpass_cutoff_hz > 0 and self.highpass_cutoff_hz < nyquist:
            self.sos_hp = signal.butter(
                N=max(1, self.highpass_order),
                Wn=self.highpass_cutoff_hz,
                btype="high",
                fs=self.sr,
                output="sos"
            )
        else:
            self.sos_hp = None

        # Build Reverb if pedalboard is available and wet_level > 0
        self.reverb_board = None
        if pedalboard is not None and self.reverb_wet_level > 0:
            try:
                # Pedalboard's JUCE Freeverb plugin sums stereo output channels (L+R)
                # when processing a 1D mono array, doubling the gain (2.0x / +6 dB).
                # To maintain true unity gain (0 dB), scale dry and wet levels by 0.5:
                wet_gain = float(self.reverb_wet_level) * 0.5
                dry_gain = max(0.0, 1.0 - float(self.reverb_wet_level)) * 0.5
                self.reverb_board = pedalboard.Pedalboard([
                    pedalboard.Reverb(
                        room_size=self.reverb_room_size,
                        damping=0.5,
                        wet_level=wet_gain,
                        dry_level=dry_gain
                    )
                ])
            except Exception:
                self.reverb_board = None

    def generate_optical_noise_and_hum(self, num_samples: int) -> np.ndarray:
        """
        Synthesizes realistic, subtle 1960s optical film surface hiss and faint projector AC hum.
        """
        # If external noise file is specified and exists, load random crop
        if self.noise_file and os.path.exists(self.noise_file) and sf is not None:
            try:
                info = sf.info(self.noise_file)
                if info.frames >= num_samples:
                    start_f = random.randint(0, info.frames - num_samples)
                    noise, _ = sf.read(self.noise_file, start=start_f, frames=num_samples, dtype="float32")
                    if noise.ndim > 1:
                        noise = np.mean(noise, axis=-1)
                    noise_std = np.std(noise)
                    if noise_std > 1e-6:
                        return (noise / noise_std).astype(np.float32)
            except Exception:
                pass

        # 1. Projector AC mains hum (very subtle 50 Hz fundamental + 100 Hz octave)
        t = np.arange(num_samples) / self.sr
        f0 = self.hum_frequency_hz
        hum = (
            0.75 * np.sin(2 * np.pi * f0 * t) +
            0.25 * np.sin(2 * np.pi * (2 * f0) * t)
        ).astype(np.float32)
        hum_std = np.std(hum)
        if hum_std > 1e-6:
            hum = hum / hum_std

        # 2. Continuous optical film grain hiss (gentle warm pink noise)
        white = np.random.normal(0, 1, num_samples).astype(np.float32)
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1.0, -2.494956002, 2.017265875, -0.522189400]
        try:
            hiss = signal.lfilter(b, a, white)
        except Exception:
            hiss = white
        hiss_std = np.std(hiss)
        if hiss_std > 1e-6:
            hiss = hiss / hiss_std

        # 3. Very sparse, gentle dust ticks (controlled by crackle_probability)
        crackle = np.zeros(num_samples, dtype=np.float32)
        if self.crackle_probability > 0.0 and random.random() < self.crackle_probability:
            num_crackles = random.randint(1, 2)
            indices = np.random.randint(0, num_samples, num_crackles)
            crackle[indices] = np.random.uniform(-0.02, 0.02, num_crackles)

        # Blend components (hum_level is small, e.g. 0.005)
        h_weight = min(max(self.hum_level, 0.0), 0.5)
        combined = (1.0 - h_weight) * hiss + h_weight * hum + crackle
        combined_std = np.std(combined)
        if combined_std > 1e-6:
            combined = combined / combined_std
        return combined.astype(np.float32)

    def apply_saturation(self, audio: np.ndarray) -> np.ndarray:
        """
        Applies non-linear tanh saturation simulating 35mm optical emulsion print compression.
        Preserves unity gain at small/nominal amplitudes while smoothly soft-clipping dynamic peaks.
        """
        if self.saturation_drive <= 1.0:
            return audio
        drive = float(self.saturation_drive)
        # tanh(drive * x) / drive maintains unity derivative at zero amplitude (no artificial volume boost)
        return np.tanh(drive * audio) / drive

    def mix_at_snr(self, audio: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
        """
        Scales and mixes noise with audio at a target Signal-to-Noise Ratio (in dB).
        """
        sig_power = float(np.mean(audio ** 2))
        if sig_power < 1e-7:
            # For near-silent clips, inject a very faint ambient bed (-55 dBFS)
            faint_target = 10.0 ** (-(snr_db + 15.0) / 10.0)
            noise_power = float(np.mean(noise ** 2))
            scale = np.sqrt(faint_target / (noise_power + 1e-9))
            return audio + noise * scale

        noise_power = float(np.mean(noise ** 2))
        if noise_power < 1e-9:
            return audio

        target_noise_power = sig_power / (10.0 ** (snr_db / 10.0))
        scale = np.sqrt(target_noise_power / (noise_power + 1e-9))
        return audio + noise * scale

    def apply(self, audio_array: np.ndarray) -> np.ndarray:
        """
        Applies the complete optical sound channel degradation pipeline.
        Noise is mixed with audio and filtered through the optical sound head
        so it blends naturally into the audio bed without harsh digital high frequencies.
        """
        audio = np.asarray(audio_array, dtype=np.float32)

        # 1. Soundstage Spatial Reverb (subtle room acoustics)
        if self.reverb_board is not None:
            try:
                audio = self.reverb_board(audio, self.sr)
            except Exception:
                pass

        # 2. Mild Non-linear Tanh Saturation (Optical Emulsion Compression)
        audio = self.apply_saturation(audio)

        # 3. Optical Film Noise & Projector Hum (Mixed at target subtle SNR)
        target_snr = random.uniform(self.snr_min_db, self.snr_max_db)
        noise = self.generate_optical_noise_and_hum(len(audio))
        mixed = self.mix_at_snr(audio, noise, target_snr)

        # 4. Low-Pass Filter (Optical sound head rolloff applied to entire mix)
        if self.sos_lp is not None:
            mixed = signal.sosfilt(self.sos_lp, mixed)

        # 5. High-Pass Filter (Subsonic rumble removal)
        if self.sos_hp is not None:
            mixed = signal.sosfilt(self.sos_hp, mixed)

        # 6. Energy preservation (prevent artificial loudness inflation)
        in_rms = float(np.sqrt(np.mean(audio_array ** 2)))
        if in_rms > 1e-4:
            out_rms = float(np.sqrt(np.mean(mixed ** 2)))
            if out_rms > in_rms:
                mixed = mixed * (in_rms / out_rms)

        # 7. Peak Limiting
        mixed = np.clip(mixed, -1.0, 1.0)
        return mixed.astype(np.float32)
