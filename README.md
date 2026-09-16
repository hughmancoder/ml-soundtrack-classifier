# ML Soundtrack Classifier

An End-to-End ML Pipeline for Soundtrack Annotation with to classify and localize traditional chinese and western instruments in early film audio for film analysis. 

Focus films
- Come drink with me
- A touch of zen
- Dragon Inn

## Installation

1. Clone the repository and navigate to the root directory:
   ```bash
   git clone <repo_url>
   cd ML_soundtrack_classifier
   ```

2. (Optional but recommended) Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install the package locally (optional):
   ```bash
   `pip install -e .`
   ```

## Pipeline Architectures & Usage

The repository supports three complementary model architectures tailored for archival film soundtrack analysis:
1. **Audio Spectrogram Transformer (AST) & Foundation Model Probing** (MERT-v1-95M + LAION-CLAP fused embeddings, Apple Silicon M4 optimized)
2. **Archival CRNN** (2-channel Harmonic-Percussive CQT front-end with bidirectional GRU)
3. **CNN Baseline** (ResNet-50 Transfer Learning on CQT spectrograms)

---

### 1. Audio Spectrogram Transformer (AST) & Foundation Model Pipeline

A 7-phase end-to-end operational pipeline engineered for Apple Silicon (MPS unified memory) combining frozen foundation audio transformers (`MERT-v1-95M` at 24 kHz and `LAION-CLAP` at 48 kHz) with a 2-layer MLP Probing Head ($1,280 \rightarrow 512 \rightarrow 256 \rightarrow 12$) and dual-threshold hysteresis post-processing.

#### Train the Transformer Probing Head:
```bash
# Run the complete Phase 1-4 training pipeline (Standardize -> Slicing & Mixup Augment -> MERT+CLAP Extraction -> 5-Fold GroupKFold Training):
make train-ast

# Or run via Python directly with full data preparation:
PYTHONPATH=src venv/bin/python src/train_ast.py --full-prep

# Or train only the probing head on pre-computed feature embeddings:
PYTHONPATH=src venv/bin/python src/train_ast.py --epochs 40 --batch-size 64
```

#### Test & Continuous Film Scoring:
```bash
# Run the AST test suite (unit tests for all 7 phases):
make test-ast

# Run continuous scoring and dual-threshold hysteresis annotation on the default film (Come Drink with Me):
make ast-pipeline

# Run continuous scoring on a custom film or audio file:
make ast-pipeline input="path/to/film_audio.wav"

# Run individual pipeline phases via CLI:
PYTHONPATH=src venv/bin/python scripts/cli/ast_pipeline.py --phase 1    # Phase 1: Audio Standardization (24k/48k SoXR)
PYTHONPATH=src venv/bin/python scripts/cli/ast_pipeline.py --phase 2    # Phase 2: Slicing & Polyphonic Mixup Augmentation
PYTHONPATH=src venv/bin/python scripts/cli/ast_pipeline.py --phase 3    # Phase 3: Frozen Feature Extraction (MERT + CLAP)
PYTHONPATH=src venv/bin/python scripts/cli/ast_pipeline.py --phase 4    # Phase 4: Probing Head Training (GroupKFold)
PYTHONPATH=src venv/bin/python scripts/cli/ast_pipeline.py path/to/film.wav --phase score  # Phases 5-7: Separation, Scoring, Postprocessing
```

---

### 2. Archival CRNN Pipeline (2-Channel HPSS + CQT)

Designed specifically for noisy 1960s optical film soundtracks. Employs a 2-channel front-end (Channel 0: Harmonic for pitched sustained instruments; Channel 1: Percussive for sharp transients), frequency pooling preserving time resolution, and a bidirectional GRU tracking musical phrasing.

#### Precompute Feature Tensors (Recommended for 5-10x Faster Training):
```bash
# Pre-computes and caches 2-channel CQT+HPSS tensors into data/features_cqt_hpss/:
make precompute-features
```

#### Train the Archival CRNN:
```bash
# Train the CRNN with dynamic 6-8s crops and optical degradation simulation:
make train-rnn

# Resume training from an existing checkpoint:
make resume-rnn
```

#### Test & Evaluate the Archival CRNN:
```bash
# Evaluate CRNN against ground-truth film cue sheet annotations:
make test-rnn

# Run streaming 10s Hanning-overlap sliding window inference on an audio file:
make rnn-pipeline input="data/test/come-drink-with-me/Come Drink With Me.wav"
```

---

### 3. CNN Baseline Pipeline (ResNet-50 Transfer Learning)

Adapts an ImageNet pre-trained ResNet-50 backbone to accept 1-channel CQT spectrograms with SpecAugment and class-weighted loss.

#### Train the CNN Baseline:
```bash
# Start CNN training:
make train-cnn

# Resume CNN training from checkpoint:
make resume-cnn
```

#### Test & Evaluate the CNN Baseline:
```bash
# Evaluate CNN model against cue sheet annotations:
make test-cnn

# Run sliding window inference on continuous audio:
make cnn-pipeline input="path/to/film.wav"
```

---

### Graphical User Interface (GUI)

Launch the standalone desktop application with interactive drag-and-drop inference, audio playback, waveform scrubbing, and event export:

```bash
make gui
```

### Instruments in Focus

[training dataset README](data/train/README.md)

train_labels:
  - sheng
  - dizi
  - erhu
  - pipa
  - suona
  - guzheng
  - yangqin
  - percussion 
  - strings
  - woodwind  
  - brass 

# TODO

- [ ] Package rnn model and upload to google drive