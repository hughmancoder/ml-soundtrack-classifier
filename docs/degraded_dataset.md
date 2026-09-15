## Synthetic Degradation Pipeline

**Implementation File:** [`src/soundtrack_classifier/ingestion/degradation.py`](file:///Users/hughsignoriello/Developer/ml-soundtrack-classifier/src/soundtrack_classifier/ingestion/degradation.py)

The synthetic degradation pipeline (`SyntheticDegradation` class) applies a sequential chain of digital signal processing nodes using the `pedalboard` library to the clean instrument samples:

1. **Spatial Simulation (Algorithmic Reverb):** The Shaw Brothers films were notoriously recorded on highly reflective, cavernous soundstages. Using `pedalboard.Reverb`, algorithms mimicking large, untreated acoustic spaces are applied to the dry instrument samples, introducing temporal smearing and early reflections.
   * *Parameters:* `room_size=0.8`, `damping=0.5`, `wet_level=0.3`, `dry_level=0.7`

2. **Dynamic Range Compression:** The `pedalboard.Compressor` and `pedalboard.Limiter` nodes aggressively flatten the dynamic range, mimicking the technological limitations of 1960s magnetic and optical recording tape.
   * *Parameters (Compressor):* `threshold_db=-20`, `ratio=4`, `attack_ms=5`, `release_ms=50`
   * *Parameters (Limiter):* `threshold_db=-1.0`

3. **Frequency Bandlimiting:** Utilizing `pedalboard.LowpassFilter` and `pedalboard.HighpassFilter`, the audio is bandlimited to emulate the narrow frequency response of vintage optical film heads.
   * *Parameters:* Highpass `cutoff_frequency_hz=100.0`, Lowpass `cutoff_frequency_hz=8000.0`

4. **Lossy Degradation:** The `pedalboard.GSMFullRateCompressor` and `pedalboard.Bitcrush` plugins introduce quantization noise and harmonic distortion inherent to legacy analog-to-digital transfers.
   * *Parameters (Bitcrush):* `bit_depth=12`

5. **Noise Injection:** Finally, synthetic broadband pink noise and sparse vinyl crackle are computationally overlaid onto the signal to replicate the continuous optical film rumble present throughout the *Come Drink with Me* runtime.
   * *Parameters:* Pink noise floor normalized to roughly `-30dB`, with random crackle spikes injected at a `0.1%` probability per sample normalized to `-20dB`. The mixed signal is strictly clipped between `-1.0` and `1.0`.