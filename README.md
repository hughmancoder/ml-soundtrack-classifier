# ML Soundtrack Classifier

An End-to-End ML Pipeline for Soundtrack Annotation that combines Music Source Separation (MSS) with Sound Event Detection (SED) to classify and localize traditional instruments in film audio.

## Installation

1. Clone the repository and navigate to the root directory:
   ```bash
   git clone <repo_url>
   cd ML_soundtrack_classifier
   ```

2. (Optional but recommended) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
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

### Instruments in Focus

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

  ### Datasets

  ChMusic
  MUSDB18
  MedleyDB-Instruments
