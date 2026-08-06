Stage 1: Ingestion & Pre-Processing
Audio Extraction: Extract the mono or stereo audio track from the legacy film file.

Resampling & Normalization: Standardize the audio to a uniform sample rate (e.g., 32 kHz or 44.1 kHz) and normalize the amplitude.

Denoising (Optional but Recommended): Apply a lightweight pass to reduce broadband optical film rumble before feeding it to the heavier neural networks.

Stage 2: Source Separation (Unbaking the Mix)
Model: Fine-tuned BS-RoFormer or HTDemucs.

Process: The polyphonic, noisy film audio is passed through the separator.

Output: The model outputs multiple isolated audio stems (e.g., one audio file for string instruments, one for percussion, one for wind instruments, and one for noise/dialogue to be discarded).

Stage 3: Feature Extraction
Conversion: The isolated audio stems are converted from raw waveforms into 2D visual representations of sound.

Algorithm: Constant-Q Transform (CQT).

Why CQT? Unlike standard Mel-spectrograms, CQT provides a logarithmic frequency scale that perfectly aligns with musical pitches, granting higher resolution to the intricate harmonic structures and pitch-bends typical of instruments like the Erhu.

Stage 4: Sound Event Detection (Classification & Localization)
Model: Hierarchical Token-Semantic Audio Transformer (HTS-AT) or Patchout Audio Spectrogram Transformer (PaSST).

Process: The CQT spectrograms are chopped into "patches" and fed into the Transformer. The model evaluates the temporal and spectral features to identify what instrument is playing and exactly when.
◊
Optimization: During training, Asymmetric Focal Loss is used to penalize the model for missing rare instruments (like a brief cymbal crash) while preventing it from over-predicting dominant, continuously playing instruments.

Output: Continuous, frame-level probability maps (e.g., a graph showing a 95% probability of a Guzheng playing at frame 1042).

Stage 5: Post-Processing & TimestampingÇ◊
Technique: Sound Event Bounding Boxes (SEBB) or Median Filtering.

Process: The raw probability maps from the Transformer are smoothed out. Heuristic thresholds (e.g., "ignore any sound under 0.5 seconds") are applied to prevent the system from generating rapid, jittery on/off timestamps.

Output: Definitive, discrete start and end timestamps for each instrument class.

Stage 6: Annotation Output
Formatting: The final timestamps and class labels are compiled into a structured format (JSON, XML, or a standard subtitle format like SRT).

Result: A machine-readable, human-verifiable log of exactly when and where every instrument played throughout the film.