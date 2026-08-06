# 1. The Broad Stem Mapping Strategy

You should configure your source separator to output **five broad acoustic stems**. You group the instruments based on their physical acoustic properties (how the sound is generated), which makes it much easier for the separator's neural network to learn the distinct spectrogram patterns.

| Separator Stem | Acoustic Profile | Target `train_labels` Included |
| :--- | :--- | :--- |
| **1. Plucked Strings** | Sharp transient attacks, rapid decay. | `pipa`, `guzheng`, `yangqin` |
| **2. Bowed Strings** | Sustained harmonic tones, heavy vibrato. | `erhu`, `strings` |
| **3. Winds & Brass** | Breath noise, sustained reed/air column harmonics. | `dizi`, `xiao`, `sheng`, `suona`, `woodwind`, `brass` |
| **4. Percussion** | Broadband, impulsive, mostly non-harmonic. | `percussion` |
| **5. Residual** | Everything else (to be discarded). | Dialogue, foley, optical film rumble, tape hiss. |