# Soundtrack Classifier Evaluation Results

The following table summarizes the sliding-window inference performance of the trained models against the ground truth cue sheet annotations for the film *Come Drink With Me* (1966, Shaw Brothers, 94 minutes, 5,684 evaluation windows).

| Model | Top-1 Music Accuracy | Top-3 Music Accuracy | Non-Music Accuracy (Rejections) | Overall Film Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **ResNet-18 CNN Baseline** | 10.19% | 71.45% | **84.57%** | 54.28% |
| **ResNet-50 CNN (Transfer Learning)** | 22.85% | 85.12% | 78.10% | **55.30%** |
| **Archival CRNN (2-Ch HPSS + Bi-GRU)** | **30.04%** | **86.16%** | 66.39% | 51.48% |
| **AST (MERT + CLAP Foundation Probing)** | 26.98% | 72.98% | 57.52% | 44.92% |

---

## Key Insights

1. **Top Music Discriminator**:
   - The **Archival CRNN** maintains the highest **Top-1 Music Accuracy (30.04%)** and **Top-3 Music Accuracy (86.16%)**, benefiting from sequential recurrent modeling of temporal phrasing and sustain dynamics across its dual-channel (Harmonic + Percussive) CQT front-end.

2. **Best Overall Film Performance**:
   - The **ResNet-50 CNN (Transfer Learning)** achieves the highest **Overall Film Accuracy (55.30%)** across all 5,684 evaluation windows, effectively balancing music identification with robust non-music rejection (**78.10%**).
   - The **ResNet-50 CNN** achieves more than **2.2x higher Top-1 accuracy** (22.85% vs. 10.19%) and a **13.67 percentage-point gain in Top-3 accuracy** (85.12% vs. 71.45%) over the lightweight ResNet-18 baseline.

3. **Foundation Model Probing (AST: MERT + CLAP)**:
   - The **AST Probing Model** demonstrates competitive zero-shot / frozen backbone capability, reaching **26.98% Top-1 Music Accuracy** (outperforming the ResNet-50 CNN on musical cues: 26.98% vs. 22.85%) and **72.98% Top-3 Music Accuracy**.
   - By freezing MERT (24 kHz) and CLAP (48 kHz) weights and only training an 800K parameter MLP Probing Head, AST extracts rich timbral features (particularly strong on dizi, sheng, and percussion) without expensive end-to-end transformer training.
   - Non-music accuracy is currently lower (57.52%) than the CNN baselines because the probing head was trained primarily on isolated instrument clips; integrating full end-to-end LoRA/adapter fine-tuning or threshold calibration in Phase 7 is projected to elevate overall film accuracy further.

4. **Baseline Trade-Offs**:
   - The **ResNet-18 CNN Baseline** maintains strong non-music rejections (**84.57%**), but suffers on individual instrument recognition (10.19% Top-1) compared to deeper architectures.
