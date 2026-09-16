# Soundtrack Classifier Evaluation Results

The following table summarizes the sliding-window inference performance of the trained models against the ground truth cue sheet annotations for the film *Come Drink With Me*.

| Model | Top-1 Music Accuracy | Top-3 Music Accuracy | Non-Music Accuracy (Rejections) | Overall Film Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **ResNet-18 CNN Baseline** | 10.19% | 71.45% | **84.57%** | 54.28% |
| **ResNet-50 CNN (Transfer Learning)** | 22.85% | 85.12% | 78.10% | **55.30%** |
| **Archival CRNN** | **30.04%** | **86.16%** | 66.39% | 51.48% |

## Key Insights
- The **ResNet-50 CNN (Transfer Learning)** achieves the highest **Overall Film Accuracy (55.30%)** across all 5,684 evaluation windows, effectively balancing music identification with robust non-music rejection (**78.10%**).
- The **Archival CRNN** maintains the highest **Top-1 Music Accuracy (30.04%)** and **Top-3 Music Accuracy (86.16%)**, benefiting from sequential recurrent modeling of temporal phrasing and sustain dynamics.
- The **ResNet-50 CNN** achieves more than **2.2x higher Top-1 accuracy** (22.85% vs. 10.19%) and a **13.67 percentage-point gain in Top-3 accuracy** (85.12% vs. 71.45%) over the lightweight ResNet-18 baseline.
- The **ResNet-18 CNN Baseline** maintains strong non-music rejections (**84.57%**), but suffers on individual instrument recognition compared to deeper architectures.


