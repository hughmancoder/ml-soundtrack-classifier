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

## Usage (Phase 2 Full Pipeline)

You can run the entire end-to-end pipeline (Ingestion -> Denoising -> Source Separation -> CQT Extraction -> SED Inference) directly from the command line:

```bash
python src/soundtrack_classifier/pipeline.py path/to/your/legacy_film.mp4
```

By default, this will:
1. Extract and process the audio into `data/processed/audio/`
2. Separate the tracks into `data/processed/stems/`
3. Save CQT spectrograms to `data/processed/features/`
4. Run inference through the PaSST model and print the probability tensor shapes.

You can override the data directory by passing `--data_dir custom/path`.

## Usage (CNN Local Pipeline)

The CNN pipeline skips source separation and uses a lightweight ResNet-18 model optimized for Apple Silicon (MPS). 

To train the CNN classifier:
```bash
python scripts/train_cnn.py
```
*(Ensure your pre-sliced 3s audio chunks are located in `data/train/<class_name>/`)*

To run sliding-window inference on a continuous audio file using the trained CNN model:
```bash
python scripts/cnn_pipeline.py path/to/your/legacy_film.wav
```

### Graphical User Interface (GUI)

You can launch a professional desktop application with a drag-and-drop interface, waveform visualizer, and results table to run sliding window inference visually!

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

A CRNN (Convolutional Recurrent Neural Network) combining a lightweight 2D residual CNN backbone with a 2-layer Bidirectional GRU (BiGRU) is the most robust and practical starting point for polyphonic instrument detection and Sound Event Detection (SED) on archival audio.

. Audio Pre-ProcessingSampling Rate: Downsample to 16 kHz or 22.05 kHz. 1960s 35mm optical audio rolls off sharply above 5–6 kHz. Retaining higher rates merely feeds optical hiss into the model.Peak Normalisation: Normalise audio chunks to $-1.0\text{ dBFS}$ to mitigate dynamic range compression swings across film reels.2. Spectrogram Representation: Constant-Q Transform (CQT)Transform: CQT using nnAudio.features.cqt.CQT2010v2 or torchaudio. Because it runs as a 1D convolutional layer inside PyTorch, you can compute it on the fly directly on GPU/MPS.$f_{\min}$ and $f_{\max}$: $f_{\min} = 65.4\text{ Hz}$ ($C_2$), covering low string/drum resonances; $f_{\max} \approx 6.5\text{ kHz}$.Bins per Octave: 24 to 36 bins/octave (2–3 bins per semitone) across 6 to 7 octaves (~144 to 216 total frequency bins). This provides the pitch resolution required to separate microtonal pitch inflections (rouxian, portamento) from fixed-pitch Western accompaniment.Dynamic Range: Log-amplitude scaling: $\log(1 + 10 \cdot \vert{}X_{\text{CQT}}\vert{})$ or $20 \cdot \log_{10}(\vert{}X_{\text{CQT}}\vert{} + \epsilon)$.