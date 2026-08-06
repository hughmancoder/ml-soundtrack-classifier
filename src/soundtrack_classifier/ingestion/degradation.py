import numpy as np
import pedalboard

class SyntheticDegradation:
    """
    Applies synthetic audio degradation using `pedalboard` to mimic legacy 1960s 
    Shaw Brothers optical film tracks.
    """
    def __init__(self, sr=44100):
        self.sr = sr
        
        # Define the digital signal processing chain
        self.board = pedalboard.Pedalboard([
            # 1. Spatial Simulation (Algorithmic Reverb)
            # Simulates highly reflective cavernous soundstages.
            pedalboard.Reverb(room_size=0.8, damping=0.5, wet_level=0.3, dry_level=0.7),
            
            # 2. Dynamic Range Compression
            # Mimics limitations of 1960s magnetic/optical tape.
            pedalboard.Compressor(threshold_db=-20, ratio=4, attack_ms=5, release_ms=50),
            
            # 3. Frequency Bandlimiting
            # Emulates narrow frequency response of vintage optical film heads.
            pedalboard.HighpassFilter(cutoff_frequency_hz=100.0),
            pedalboard.LowpassFilter(cutoff_frequency_hz=8000.0),
            
            # 4. Lossy Degradation
            # Introduces quantization noise and harmonic distortion.
            pedalboard.GSMFullRateCompressor(),
            pedalboard.Bitcrush(bit_depth=12),
            
            # 5. Final Limiter to catch peaks
            pedalboard.Limiter(threshold_db=-1.0)
        ])

    def generate_vinyl_noise(self, num_samples):
        """
        Synthesizes a continuous optical film rumble (pink-ish noise + crackle).
        Since we don't have a specific vinyl crackle file yet, we use algorithmic noise.
        """
        # Generate white noise
        white_noise = np.random.normal(0, 1, num_samples)
        
        # Simple pink-ish filter (1/f) approximation using a lowpass
        b, a = [0.049922035, -0.095993537, 0.050612699, -0.004408786], [1, -2.494956002, 2.017265875, -0.522189400]
        try:
            from scipy.signal import lfilter
            pink_noise = lfilter(b, a, white_noise)
        except ImportError:
            # Fallback to white noise if scipy isn't loaded correctly
            pink_noise = white_noise

        # Normalize
        pink_noise = pink_noise / np.max(np.abs(pink_noise))
        
        # Synthesize sparse "crackle" (random high-amplitude spikes)
        crackle = np.zeros(num_samples)
        num_crackles = int(num_samples * 0.001)  # 0.1% chance per sample
        crackle_indices = np.random.randint(0, num_samples, num_crackles)
        crackle[crackle_indices] = np.random.uniform(-1, 1, num_crackles)
        
        # Combine noise floor (-30dB) and crackle (-20dB)
        noise_profile = (pink_noise * 0.0316) + (crackle * 0.1)
        return noise_profile

    def apply(self, audio_array):
        """
        Applies the full degradation pipeline to a 1D numpy array audio signal.
        """
        # 1. Apply the pedalboard chain
        degraded = self.board(audio_array, self.sr)
        
        # 2. Inject synthetic noise
        noise = self.generate_vinyl_noise(len(degraded))
        
        # Mix the degraded signal with the noise
        mixed = degraded + noise
        
        # Ensure we don't clip above 1.0 or below -1.0 after mixing
        mixed = np.clip(mixed, -1.0, 1.0)
        
        return mixed
