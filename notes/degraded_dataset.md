## Synthetic Degradation Pipeline

The synthetic degradation pipeline applies a sequential chain of digital signal processing nodes to the clean instrument samples:

1. **Spatial Simulation (Convolutional Reverb):** The Shaw Brothers films were notoriously recorded on highly reflective, cavernous soundstages. Using `pedalboard.Convolution`, specific impulse responses (IRs) mimicking large, untreated acoustic spaces are applied to the dry instrument samples, introducing the requisite temporal smearing and early reflections.

2. **Dynamic Range Compression:** The `pedalboard.Compressor` and `pedalboard.Limiter` nodes are utilized to aggressively flatten the dynamic range, mimicking the technological limitations of 1960s magnetic and optical recording tape.

3. **Frequency Bandlimiting:** Utilizing `pedalboard.LowpassFilter` and `pedalboard.HighpassFilter`, the audio is bandlimited (typically cutting frequencies below 100 Hz and above 8,000 Hz) to emulate the narrow frequency response of vintage optical film heads.

4. **Lossy Degradation:** The `pedalboard.GSMFullRateCompressor` and `pedalboard.Bitcrush` plugins introduce subtle quantization noise and harmonic distortion inherent to legacy analog-to-digital transfers.

5. **Noise Injection:** Finally, synthetic broadband pink noise and vinyl crackle are computationally overlaid onto the signal to replicate the continuous optical film rumble present throughout the *Come Drink with Me* runtime.