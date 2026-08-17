# MVP Pipeline Architecture & Workflow

## 1. Data Preparation (Pure Slicing)
Instead of complex synthetic mixing, the MVP relies on clean, isolated audio samples to train a robust baseline classifier.
- **Pre-Slicing**: Use Python (librosa or pydub) to slice pure recordings of the target labels (sheng, dizi, erhu, percussion, etc.) into exact 3-second chunks.
- **Energy Thresholding**: Implement an RMS energy check during slicing to automatically discard silent or near-silent clips.
- **Storage Management**: Delete the large original .wav files after slicing to preserve the 20-30GB disk limit. The dataset requires simple folder-based structuring (e.g., `/data/train/erhu`).

## 2. Source Separation (SKIPPED)
- **Status for MVP**: Bypassed.
- **Rationale**: Because the goal of this baseline is to predict the single dominant instrument in a given timeframe (Multi-Class Classification), "unbaking" the mix is unnecessary. The neural network will learn to identify the loudest harmonic or percussive signature in the window. (Source separation will be reintroduced in Phase 2 for full polyphonic detection).

## 3. Feature Extraction (On-The-Fly CQT)
Convert audio waveforms into 2D visual representations for the neural network dynamically to save hard drive space.
- **Algorithm**: Constant-Q Transform (CQT). The CQT perfectly aligns with musical pitches and grants higher resolution to intricate harmonic structures like the vibrato of an erhu.
- **Implementation**: Do not save spectrograms to the hard drive. Compute the CQT dynamically on the Mac GPU during training using the nnAudio library inside the PyTorch DataLoader.
- **The mathematical foundation of the CQT is defined as**:
$$X[k] = \frac{1}{N[k]} \sum_{n=0}^{N[k]-1} x[n] \cdot w_k[n] e^{-j\frac{2\pi Q}{N[k]}n}$$

## 4. Dominant Instrument Classification (Stage 3)
Treat the audio classification task as an image classification problem based on the CQT spectrogram.
- **Model**: Use a lightweight, standard CNN such as ResNet-18 or MobileNetV2. (Do not use Transformers or CRNNs for this baseline).
- **Objective**: Train the network using standard Cross-Entropy Loss to output a single probability for the dominant instrument out of the 12 target classes.
- **Mac Optimization**: Run natively on Apple Silicon using the Metal Performance Shaders backend (`device="mps"`). Because the model is lightweight and clips are only 3 seconds, you can safely use standard batch sizes (e.g., 32 or 64) without Out-Of-Memory (OOM) errors.

## 5. Inference & Timestamp Generation (Stage 4)
Because the MVP uses a classifier rather than a specialized Sound Event Detection (SED) model, timestamps are generated programmatically via sliding windows.
- **The Sliding Window Technique**: During film analysis, a script slides a 3-second window across the continuous film audio track using a 1-second stride (e.g., 0–3s, 1–4s, 2–5s).
- **Post-Processing**: The CNN predicts the dominant instrument for each 3-second slice. A simple algorithmic rule merges consecutive identical predictions into a continuous timestamp (e.g., `[Erhu, Erhu, Erhu]` becomes `Erhu active: 0.0s - 5.0s`). Median filtering can be applied to smooth out single-frame misclassifications.

## 6. Training Pipeline Details (`train_cnn.py`)
- **Dataset Split**: Automatically performs an 80/20 train/validation split using `torch.utils.data.random_split` with a fixed seed for reproducibility.
- **Model Used**: `ResNet18Classifier` (a standard lightweight ResNet-18).
- **Handling Class Imbalance**: Calculates dynamic class weights based on the training split's distribution. These weights are passed to `nn.CrossEntropyLoss`, forcing the model to heavily penalize errors on underrepresented instruments.
- **Augmentation**: Applies `SpecAugment` (time and frequency masking) selectively to minority classes directly on the GPU.
- **Checkpointing & Early Stopping**: Tracks validation accuracy after every epoch. The model state is saved to `models/resnet18_cnn.pt` only if validation accuracy improves. Early stopping halts training if no improvement is seen for a set number of epochs (default 10).
- **Training Curves**: Currently, it logs training and validation loss/accuracy to the standard console output. It does *not* presently export visual curves (e.g., TensorBoard or Weights & Biases), though `wandb` could be integrated in the future for real-time visualization.

## Hardware for training

Apple M4
Memory: 16gb
