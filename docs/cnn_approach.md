# MVP Pipeline Architecture & Workflow

## 1. Data Preparation (Pure Slicing & Degradation)
Instead of complex synthetic multi-instrument mixing, the MVP relies on clean, isolated audio samples enhanced with synthetic vintage film degradation to train a robust baseline classifier.
- **Pre-Slicing**: Slices recordings of the target labels (sheng, dizi, erhu, percussion, nonmusic, etc.) into exact 3-second chunks (44.1 kHz mono) using `librosa` and `soundfile` (`slice_audio_file` in `src/soundtrack_classifier/cnn/dataset.py`, `scripts/slice_chmusic.py`, `scripts/slice_percussion.py`, and `scripts/slice_nonmusic.py`).
- **Energy Thresholding**: RMS energy checks are performed during slicing (`rms_threshold`) to automatically discard silent or near-silent clips. Additionally, `scripts/delete_silent_clips.py` cleans up any flagged silent audio.
- **Folder Structuring**: The dataset uses simple folder-based structuring (e.g., `data/train/<instrument>/`).
- **On-The-Fly Synthetic Degradation**: Implemented via `SyntheticDegradation` (`pedalboard`) in `SoundtrackCNNDataset` to emulate vintage optical film audio (algorithmic cavernous reverb, dynamic range compression/limiting, bandlimiting high/low pass filters, bitcrushing, pink noise, and vinyl crackle).
<!-- - **Storage Management (Deleting original files)**: Delete the large original .wav files after slicing to preserve the 20-30GB disk limit. (Not implemented: raw master files are retained in `data/raw/`). -->

## 2. Source Separation (SKIPPED in CNN Pipeline)
- **Status for MVP**: Bypassed.
- **Rationale**: Because the goal of this baseline is to predict the single dominant instrument in a given timeframe (Multi-Class Classification), "unbaking" the mix is unnecessary. The neural network learns to identify the loudest harmonic or percussive signature in the window.
<!-- - **Phase 2 Polyphonic Detection**: Source separation will be reintroduced in Phase 2 for full polyphonic detection. (Note: A Demucs wrapper is present in `src/soundtrack_classifier/separation/demucs_wrapper.py`, but source separation is unused during training and sliding window inference). -->

## 3. Feature Extraction (On-The-Fly CQT)
Convert audio waveforms into 2D visual representations for the neural network dynamically to save hard drive space.
- **Algorithm**: Constant-Q Transform (CQT) using `nnAudio.features.cqt.CQT1992v2` ($f_{\text{min}}=32.7\text{ Hz}$, 84 bins, 12 bins per octave). The CQT aligns with musical pitches and grants higher resolution to intricate harmonic structures like the vibrato of an erhu.
- **Implementation**: Waveforms are stored as `.wav` clips without caching spectrograms to disk. The dataset outputs raw audio tensors, and the CQT is computed dynamically in batches on the GPU (`device="mps"`) inside the training and inference loops (rather than inside DataLoader `__getitem__`), allowing multi-process worker loading (`num_workers > 0`) without GPU context errors.
- **The mathematical foundation of the CQT is defined as**:
$$X[k] = \frac{1}{N[k]} \sum_{n=0}^{N[k]-1} x[n] \cdot w_k[n] e^{-j\frac{2\pi Q}{N[k]}n}$$

## 4. Dominant Instrument Classification (Stage 3)
Treat the audio classification task as an image classification problem based on the CQT spectrogram.
- **Model**: `ResNet18Classifier` (`src/soundtrack_classifier/cnn/resnet.py`), an adapted ResNet-18 modified with a 1-channel `conv1` layer to accept single-channel spectrograms and a linear head outputting logits over the target classes.
<!-- - **Alternative Architectures**: MobileNetV2, CRNNs, or Audio Transformers (Not implemented; ResNet-18 is used exclusively). -->
- **Objective**: Train the network using `nn.CrossEntropyLoss` with inverse-frequency class weights to output the dominant instrument out of the 12 target classes (`sheng, dizi, erhu, pipa, suona, guzheng, yangqin, percussion, strings, woodwind, brass, nonmusic`).
- **Mac Optimization**: Runs natively on Apple Silicon using Metal Performance Shaders (`device="mps"`), falling back to CPU if unavailable. Standard batch size is 32.

## 5. Inference & Timestamp Generation (Stage 4)
Because the MVP uses a classifier rather than a specialized Sound Event Detection (SED) model, timestamps are generated programmatically via sliding windows (`scripts/cnn_pipeline.py` and `src/soundtrack_classifier/cnn/inference.py`).
- **The Sliding Window Technique**: Slides a 3.0-second window across continuous film audio with a 1.0-second stride (e.g., 0–3s, 1–4s, 2–5s).
- **Silence Gating**: RMS energy check skips inference or flags pure silence for windows below threshold ($\tau = 0.005$ or $0.001$), avoiding hallucinations on tape hiss.
- **Categorical Smoothing**: Categorical mode filtering (majority vote with kernel size 3) removes single-frame classification flicker without mathematical interpolation artifacts inherent to numeric median filtering.
- **Contiguous Merging & Boundary Calculation**: Consecutive identical predictions are merged into continuous chronological events with exact boundaries tied to window strides.
- **Duration Thresholding**: Purges short transient events below a minimum threshold ($\ge 1.0\text{s}$) to discard spurious predictions.
- **Desktop Graphical Interface**: Packaged into a local web GUI powered by Gradio (`src/soundtrack_classifier/gui.py`, run via `make gui`) supporting video/audio drag-and-drop, automated FFmpeg extraction, and table generation.
<!-- - **Dual-Stream Transient Detection**: Utilizing `librosa.onset.onset_detect` for sub-second percussion micro-clips (Window = 0.5s, Stride = 0.1s) merged with a melody stream (Window = 3.0s, Stride = 1.0s). (Not implemented / Planned for future improvement). -->

## 6. Training Pipeline Details (`train_cnn.py`)
- **Dataset Split**: Automatically performs an 80/20 train/validation split using `torch.utils.data.random_split` with a fixed seed (`seed=42`) for reproducibility.
- **Model Used**: `ResNet18Classifier` (a standard lightweight ResNet-18 adapted for 1-channel CQT).
- **Handling Class Imbalance**: Calculates dynamic inverse-frequency class weights based on the training split's distribution:
  $$\text{weight}_c = \frac{N_{\text{total}}}{C \times N_c}$$
  These weights are passed to `nn.CrossEntropyLoss`, penalizing errors on underrepresented instruments.
- **Augmentation**:
  - **SpecAugment**: Frequency and time masking (`torchaudio.transforms.FrequencyMasking`, `TimeMasking`) implemented in `train_cnn.py` for minority classes on GPU (configurable in `config/config.yaml`, currently toggled off).
  - **Synthetic Degradation**: On-the-fly digital signal degradation via `pedalboard` applied inside the dataset pipeline (enabled with 80% probability).
- **Checkpointing & Resuming**: Saves model weights, class mappings, optimizer state, epoch, and validation accuracy to `models/resnet18_cnn.pt` whenever validation accuracy improves. Training can be resumed using `--resume` (or `make resume-cnn`).
- **Early Stopping**: Halts training if validation accuracy fails to improve for a configurable number of epochs (`early_stopping_patience` in `config/config.yaml`, default 5).
- **Training Curves & Metrics**:
  - Training and validation loss/accuracy are logged to console and written to `models/training_metrics.csv` after every epoch.
  - Visual training curves are plotted and saved to `models/training_curves.png` using `scripts/plot_training_curves.py` (`make plot-curves`).
<!-- - **Live Experiment Tracking**: Real-time cloud dashboard tracking via TensorBoard or Weights & Biases (`wandb`). (Not implemented; CSV metrics and matplotlib script are used instead). -->

## Hardware for training

Apple M4
Memory: 16gb
