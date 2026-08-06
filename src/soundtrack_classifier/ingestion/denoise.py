import numpy as np
import scipy.signal as signal

def apply_lightweight_denoise(audio: np.ndarray, sr: int, cutoff_freq: float = 60.0) -> np.ndarray:
    """
    Applies a lightweight high-pass filter to reduce broadband optical film rumble
    common in legacy film audio.
    
    Args:
        audio: 1D numpy array containing the audio waveform.
        sr: Sample rate of the audio.
        cutoff_freq: Cutoff frequency for the high-pass filter (Hz). Default 60Hz.
                     This removes low rumble without affecting most musical pitches.
    Returns:
        Filtered audio array.
    """
    print(f"Applying lightweight denoising (high-pass filter at {cutoff_freq}Hz)...")
    
    # Nyquist frequency is half the sample rate
    nyquist = 0.5 * sr
    normalized_cutoff = cutoff_freq / nyquist
    
    # Design a 4th order Butterworth high-pass filter
    b, a = signal.butter(4, normalized_cutoff, btype='highpass')
    
    # Apply the filter using filtfilt for zero phase distortion
    filtered_audio = signal.filtfilt(b, a, audio)
    
    return filtered_audio.astype(np.float32)
