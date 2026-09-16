# End-to-End Computational Workflow

The complete software execution workflow is structured into five sequential processing modules optimized for local execution:

## 1. Processing Modules

- **Ingestion and Feature Preprocessing Module**: Ingests raw film audio, downmixes to mono at 32 kHz, normalizes signal RMS to -20 dBFS, and extracts dual 128-bin Log-Mel and 128-bin CQT tensors saved to local storage as compressed float16 arrays.
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