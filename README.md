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

## Models 

The repository supports three complementary model architectures tailored for archival film soundtrack analysis:
1. **Audio Spectrogram Transformer (AST) and Foundation Model Probing** (MERT-v1-95M + LAION-CLAP fused embeddings, Apple Silicon M4 optimized)
2. **Archival CRNN** (2-channel Harmonic-Percussive CQT front-end with bidirectional GRU)
3. **CNN Baseline** (ResNet-50 Transfer Learning on CQT spectrograms)

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

