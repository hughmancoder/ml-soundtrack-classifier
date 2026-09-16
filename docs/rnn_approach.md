# End-to-End Computational Workflow

The complete software execution workflow is structured into five sequential processing modules optimized for local execution:

## 1. Processing Modules

- **Ingestion and Feature Preprocessing Module**: Ingests raw film audio, downmixes to mono at 32 kHz (or 22.05 kHz for CRNN), normalizes signal RMS to target level, and extracts dual-channel CQT+HPSS tensors saved to local storage as compressed float16 arrays. To prevent GPU memory exhaustion and ensure complete temporal capture for extremely long archival tracks, this stage applies a **30-second sliding window chunking mechanism** before concatenating the features.
- **Model Fine-Tuning Module**: Loads pre-trained MobileNetV3/CRNN backbones, applies SpecAugment and Mixup data augmentations, and trains models on PyTorch using MPS acceleration (float16 autocast) under Asymmetric Loss optimization.
- **Sliding-Window Inference Module**: Streams long film soundtracks through the fine-tuned CRNN using 10-second windows with a 1-second step size, generating time-stamped probability matrices.
- **Post-Processing Module**: Applies class-specific binarization thresholds $\tau_k$ and adaptive median filtering $L_k$ to convert probability matrices into continuous time intervals.
- **Annotation Export Module**: Formats localized predictions into standardized analytical formats, including CSV files, structured JSON metadata, and ELAN Annotation Format (`.eaf`) tracks for filmic analysis.

## 2. Quantitative Evaluation Metrics

Model performance is measured using standard Sound Event Detection (SED) metrics:

- **Mean Average Precision (mAP)**: Evaluates global multi-label classification accuracy across all 13 target classes.
- **Segment-Based F1-Score ($F_1^{\text{seg}}$)**: Evaluates temporal localization accuracy over non-overlapping 1.0-second time intervals:
  $$P_{\text{seg}} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad R_{\text{seg}} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad F_1^{\text{seg}} = \frac{2 \cdot P_{\text{seg}} \cdot R_{\text{seg}}}{P_{\text{seg}} + R_{\text{seg}}}$$
- **Event-Based F1-Score ($F_1^{\text{event}}$)**: Measures precise onset detection within a $\pm 200\text{ ms}$ tolerance window and offset detection within $\pm 200\text{ ms}$ or 20% of event duration. This evaluates the system's ability to delineate discrete musical phrases and sound effects within complex film mixes.

---

## 3. Training 1–3s Short Audio Events (Percussion) in an 8-Second Pipeline

A key design requirement of the CRNN architecture is its **6-to-8 second temporal context window**, necessary to capture sustained melodic gestures (vibrato, slides, phrasing) across instruments like the *erhu*, *dizi*, and *pipa*. However, training discrete, short-duration events—such as 1–3 second percussion strikes (gongs, luo, bangu, drums)—within this longer window introduces a critical acoustic hazard if not handled properly.

### 3.1 The "Tiling Trap"
In naive audio padding pipelines, short clips are looped to fill the target window length:

```python
# The acoustic trap:
repeat_factor = int(np.ceil(self.num_samples / max(1, len(audio))))
tiled = np.tile(audio, repeat_factor)
return tiled[:self.num_samples]
```

* **Acoustic Consequence**: If a 1.5-second clip of an isolated gong strike or drum hit is repeated 5 to 6 times across 8 seconds, it creates an artificial, high-tempo rhythmic loop (120–180 BPM).
* **Network Hallucination**: The Bidirectional GRU (Bi-GRU) recurrent layers possess temporal memory cells specifically designed to track rhythmic and sequential dependencies. Under periodic tiling, the recurrent weights learn to associate percussion with **high-frequency periodic tempo loops** rather than the genuine physical acoustics: the sharp transient attack, spectral spread, and exponential decay envelope of the strike.

### 3.2 The Solution: Random Offset Placement
Instead of looping, short events are placed onto an 8-second silent canvas at a randomly chosen temporal offset:

```python
def _pad_short_audio(self, audio: np.ndarray) -> np.ndarray:
    """Inserts a short event at a random position within the 8s silent canvas."""
    if len(audio) >= self.num_samples:
        return audio

    # 1. Create an 8-second silent buffer
    canvas = np.zeros(self.num_samples, dtype=np.float32)

    # 2. Pick a random start point so the attack happens at varying temporal positions
    max_start = self.num_samples - len(audio)
    start_idx = random.randint(0, max_start) if self.random_crop else max_start // 2

    canvas[start_idx : start_idx + len(audio)] = audio
    return canvas
```

### 3.3 Theoretical & Practical Advantages
1. **Onset Position Invariance**: The Bi-GRU learns to detect the percussion event regardless of whether the onset occurs near the start, center, or tail of the sliding window buffer.
2. **Envelope & Timbre Fidelity**: Isolates the true acoustic signature—a singular energetic transient attack followed by a realistic ring-out decay into ambient silence or optical noise.
3. **Robust Background Separation**: When synthetic optical noise (surface hiss, crackle, optical hum) is added during training, the network learns to detect a solitary percussion strike rising above steady-state noise, directly mirroring actual film soundtrack conditions.

---

## 4. Harmonic-Percussive Source Separation (HPSS) Front-End

### 4.1 Acoustic Motivation: The Archival Film Mix Problem
In archival martial arts soundtracks (e.g., 1960s King Hu and Shaw Brothers films), musical cues frequently overlap with dense sound effects and percussion:
* **Explosive, broadband transient events**: Peking opera percussion (*luo* gongs, *bangu* frame drum, wooden clappers), weapon strikes, sword clashes, and athletic foley impacts.
* **Sustained, expressive harmonic gestures**: Bowed strings (*erhu* with slow vibrato and microtonal *huayin* slides), transverse flutes (*dizi* breath modulations), mouth organs (*sheng* chords), and modal vocal melodies.
* **Optical soundtrack degradations**: Continuous optical film hiss, hum, and surface crackle.

When a standard single-channel spectrogram (STFT or CQT) is processed by a 2D CNN, high-amplitude percussive transients spread broadband energy across all frequency bins in a single time frame. This transient blast creates extreme local gradients that can "blind" convolutional filters, obscuring subtle harmonic trajectories (*erhu* melodies or *dizi* overtones) occurring in the exact same temporal window. Conversely, strong stationary hums or bowed sustained tones can mask low-energy percussive onsets.

**Harmonic-Percussive Source Separation (HPSS)** decouples these fundamentally distinct acoustic behaviors into two dedicated spatial channels **before** feature representation enters the neural network.

---

### 4.2 Spectrogram Geometry & Anisotropic Median Filtering
In time-frequency representations (such as the Constant-Q Transform), harmonic and percussive phenomena exhibit orthogonal geometric signatures:

| Acoustic Phenomenon | Time-Frequency Geometry | Dominant Orientation |
| :--- | :--- | :--- |
| **Harmonic Components** (*erhu*, *dizi*, sustained vocal/brass) | Continuous horizontal ridges across time at stable or slowly modulating fundamental and harmonic bins. | **Horizontal ($\Delta t \gg \Delta f$)** |
| **Percussive Components** (*gongs*, *drums*, clappers, sword clashes) | Abrupt vertical stripes cutting instantaneously across broad frequency bands. | **Vertical ($\Delta f \gg \Delta t$)** |

HPSS extracts these components directly in the spectrogram domain using **anisotropic 1D median filters** (FitzGerald, 2010):

1. **Harmonic Estimation ($H_{\text{raw}}$)**:
   A median filter of temporal length $k_t$ (e.g., $k_t = 31$ frames $\approx 360\text{ ms}$) is applied horizontally across the **time** axis for every frequency bin $f$:
   $$H_{\text{raw}}[f, t] = \text{median}\Big(S[f, t - \lfloor k_t/2 \rfloor], \dots, S[f, t + \lfloor k_t/2 \rfloor]\Big)$$
   *Since percussive strikes exist only for 1–3 frames ($<35\text{ ms}$), a horizontal median filter of 31 frames completely rejects them as transient outliers, preserving only sustained horizontal lines.*

2. **Percussive Estimation ($P_{\text{raw}}$)**:
   A median filter of spectral length $k_f$ (e.g., $k_f = 31$ frequency bins) is applied vertically across the **frequency** axis for every time frame $t$:
   $$P_{\text{raw}}[f, t] = \text{median}\Big(S[f - \lfloor k_f/2 \rfloor, t], \dots, S[f + \lfloor k_f/2 \rfloor, t]\Big)$$
   *Harmonic partials exist as narrow, isolated horizontal spectral lines separated by several bins. A vertical median filter of 31 bins rejects isolated harmonic peaks as outliers, capturing only energy that spans many adjacent frequency bins simultaneously (broadband transients).*

---

### 4.3 Soft Wiener-Style Masking
Rather than applying hard binary thresholds (which introduce harsh phase cancellation and musical noise), soft ratio masks are derived using power-law exponentiation (default power $p = 2.0$):

$$M_H[f, t] = \frac{H_{\text{raw}}[f, t]^p}{H_{\text{raw}}[f, t]^p + P_{\text{raw}}[f, t]^p + \epsilon}$$

$$M_P[f, t] = \frac{P_{\text{raw}}[f, t]^p}{P_{\text{raw}}[f, t]^p + H_{\text{raw}}[f, t]^p + \epsilon}$$

The separated harmonic and percussive spectrogram tensors are obtained by point-wise multiplication with the original magnitude spectrogram $S$:
$$S_H[f, t] = S[f, t] \odot M_H[f, t]$$
$$S_P[f, t] = S[f, t] \odot M_P[f, t]$$

These are stacked along the channel dimension to yield a 2-channel tensor:
$$\mathbf{X}_{\text{HPSS}} \in \mathbb{R}^{B \times 2 \times F \times T}$$

Where:
* **Channel 0 ($S_H$)**: Harmonic representation, containing clean tonal contours, sustained resonance, and pitch slides (*huayin*).
* **Channel 1 ($S_P$)**: Percussive representation, containing precise attack onsets, decay impulses, and impact transients.

---

### 4.4 Neural Network Synergies in the Archival CRNN
By supplying the 2-channel tensor `(Batch, 2, 168, Time_Frames)` to the front-end CNN:
1. **Parallel Feature Extraction**: The initial $3 \times 3$ convolutions have separate kernel weights for Channel 0 and Channel 1, allowing the network to simultaneously learn:
   - Pitch contour detectors, harmonic template filters, and vibrato trackers on Channel 0.
   - Transient onset detectors, tempo/rhythm interval trackers, and decay slope filters on Channel 1.
2. **Noise Disentanglement**: Stationary optical film hum (50 Hz / 60 Hz hum and static optical line tone) naturally isolates into Channel 0 (horizontal), while random surface crackle and optical dust clicks isolate into Channel 1 (vertical). The recurrent GRU can track the continuous baseline in Channel 0 and the burst statistics in Channel 1 without cross-corrupting the acoustic classification heads.

---

### 4.5 Implementation & Hardware Optimization
In `soundtrack_classifier.features.cqt.compute_hpss_from_spec`:
1. **CPU Unfold vs. MPS Thrashing**:
   Apple Silicon MPS does not implement an optimized native kernel for strided 4D tensor `unfold` median filtering. Executing strided 31-tap sliding medians on MPS can cause execution slowdowns of up to $20\times$. The function selectively moves the 2D/3D spectrogram slice to CPU memory for sliding-window unfolding and median calculation before returning the resulting mask to the active device.
2. **Logarithmic Dynamic Range Scaling**:
   Following HPSS masking, the dual-channel features undergo dynamic range compression:
   $$\mathbf{X}_{\text{out}} = \log\left(1.0 + 10.0 \cdot \mathbf{X}_{\text{HPSS}}\right)$$
   This stabilizes gradients for low-amplitude instrumental decays (*guqin* dying resonance, quiet flute breath) against loud martial arts sound effects.