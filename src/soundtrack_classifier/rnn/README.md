# Archival Instrument CRNN

A Convolutional Recurrent Neural Network (CRNN) specialized for classifying traditional Chinese instruments in historical 1960s optical film soundtracks (e.g., King Hu / Shaw Brothers martial arts cinema).

---

## 1. Why 6 to 10-Second Windows?

- **Phrasing & Ornamentation**: Traditional Chinese instruments rely on continuous expressive techniques—such as an erhu portamento/slide (*huayin*), a sustained guqin decay, or a dizi breath cadence. These gestures typically span 2 to 4 seconds. A window of 6–10 seconds captures the preceding silence/attack, the core gesture, and the trailing decay.
- **Optical Noise Contrast**: Archival 35mm optical soundtracks contain continuous surface noise, hum, and crackle. A 6–10 second temporal context allows the recurrent GRU layers to separate steady-state noise baselines from transient musical events.
- **CQT Edge Stability**: Low-frequency filter kernels in the Constant-Q Transform require wider temporal support. Short clips (under 3–4 seconds) lose a significant portion of usable low-frequency data to boundary padding artefacts.

---

## 2. Optimal Front-End: 2-Channel HPSS + CQT

Rather than standard linear STFT or wideband Mel spectrograms (which waste capacity on high-frequency optical hiss and blur lower-mid microtonal inflections), the model uses:

1. **Constant-Q Transform (CQT)**:
   - $f_{\min} = 65.4\text{ Hz}$ ($C_2$, capturing deep percussion and low string resonances)
   - $f_{\max} \approx 6.5\text{--}8.1\text{ kHz}$ (bounded within the physical frequency limit of 1960s optical film audio)
   - $24$ bins per octave spanning $7$ octaves ($168$ frequency bins total)
   - Preserves musical pitch scales and microtonal pitch bending (*rouxian*, glissandi).

2. **Harmonic-Percussive Source Separation (HPSS) 2-Channel Stack**:
   - **Channel 0 (Harmonic)**: Isolates continuous pitched instruments (erhu, dizi, sheng, Western strings/brass).
   - **Channel 1 (Percussive)**: Isolates sharp attacks, plectrum strikes, and rhythmic elements (pipa and yangqin transients, Beijing opera gongs/clappers, and martial arts foley impacts).
   - **Shape**: `(Batch, 2, 168, Time_Frames)`.
   - Prevents loud percussive weapon clatter and foley impacts from blinding the network to concurrent melodic phrases.

---

## 3. Network Architecture

- **Frequency-Pooling CNN Encoder**:
  Four convolutional blocks with `MaxPool2d(kernel_size=(freq_pool, 1))`.
  Pools only along the frequency axis ($168 \to 84 \to 42 \to 21 \to 7$), preserving exact frame-level temporal resolution.
- **Linear Projection & Dropout**:
  Projects $128 \times 7 = 896$ dimensions to $256$.
- **Bidirectional GRU**:
  2 layers, hidden size 128 (bidirectional output: 256), tracking temporal phrasing, sustained decays, and noise baselines.
- **Dual Prediction Output**:
  - `return_frames=True`: Frame-level multi-label predictions $(B, T, \text{num\_classes})$.
  - `return_frames=False`: Temporally pooled clip predictions $(B, \text{num\_classes})$.

---

## 4. Operational Setup & Commands

### Training
Dynamically samples random 6-to-8 second crops (default: 8.0s) per epoch from continuous scenes or training clips:
```bash
# Train CRNN with dynamic 6-8s crops and on-the-fly degradation:
make train-rnn

# Resume training from checkpoint:
make resume-rnn
```

### Streaming Inference
Streams continuous audio through 10-second fixed buffers with 75% overlap (hop size of 2.5s) using a **Hanning window weighted average** to eliminate boundary discontinuities:
```bash
# Run streaming inference on any film soundtrack or audio track:
make rnn-pipeline input="data/raw/Come Drink With Me.wav"
```

### Cue Sheet Evaluation
Evaluates detected musical events against historical cue sheet annotations:
```bash
make test-rnn
```
