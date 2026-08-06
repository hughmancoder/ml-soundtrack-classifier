# Architectural & System Improvements

A comparison of the local MVP capabilities versus the scaled Phase 2 cloud deployment architecture.

| Improvement Area | MVP (Local Mac) | Phase 2 (Cloud Deployment) |
| :--- | :--- | :--- |
| **Source Separation** | Open-Unmix / Spleeter | **BS-RoFormer / HTDemucs**: State-of-the-art models that preserve complex harmonics of traditional instruments without robotic artifacts. |
| **Classification Model** | CRNN / MobileNetV2 | **HTS-AT / PaSST**: Audio Transformers maintaining frame-level temporal resolution for continuous, high-fidelity probability maps. |
| **Loss Function** | Standard Binary Cross-Entropy | **Asymmetric Focal Loss**: Penalizes missing rare percussive strikes while preventing over-prediction of dominant continuous instruments. |
| **Post-Processing** | Median Filtering | **Sound Event Bounding Boxes (SEBB)**: Bypasses heuristic thresholding to output definitive, highly precise start/end timestamps directly. |