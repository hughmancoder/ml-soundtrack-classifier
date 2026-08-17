## Inference

### Temporal Localization via Sliding Window Inference (SED)

A CNN trained on discrete 3-second clips cannot inherently generate timestamps for a continuous 90-minute film. To achieve temporal localization across the entire runtime of *Come Drink with Me*, the system employs a programmatic **Sliding Window Inference** protocol, standard in **Sound Event Detection (SED)**.

Let the total audio duration be $T$. We define a window size $W = 3.0$ seconds and a stride (hop size) $S = 1.0$ second. The start time of the $i$-th frame is given by:

$$ t_i = i \times S $$

The algorithm extracts the audio segment $[t_i, t_i + W]$, computes the Constant-Q Transform (CQT), and passes it to the ResNet-18 model to output a predicted class index $y_i \in \{0, 1, ..., C-1\}$. The window advances iteratively until the film concludes.

### Algorithmic Post-Processing

Because independent predictions over sliding windows are highly susceptible to temporal flickering (rapid, erratic switching between classes), the raw prediction array $Y = [y_0, y_1, ..., y_N]$ must be synthesized into a stable, contiguous log. 

#### 1. RMS Silence Cropping
Before evaluating a window, its Root Mean Square (RMS) energy is calculated to quantify the signal magnitude. For an audio segment containing $K$ discrete samples $x_1, x_2, ..., x_K$, the RMS energy is:

$$ RMS = \sqrt{\frac{1}{K} \sum_{k=1}^{K} x_k^2} $$

If $RMS < \tau$ (where $\tau = 0.001$), the window is classified as pure silence ($y_i = -1$) and skipped. This acts as an algorithmic noise gate, preventing the model from hallucinating instruments on background tape hiss or dead air.

#### 2. Categorical Mode Filtering
To suppress spurious, single-frame misclassifications, a rolling mode filter (majority vote) is applied across the prediction array $Y$. For a given index $i$ and kernel size $K = 3$, the smoothed prediction $\hat{y}_i$ is the statistical mode of the local neighborhood:

$$ \hat{y}_i = \text{mode}(y_{i-1}, y_i, y_{i+1}) $$

This categorical smoothing correctly suppresses noise without mathematically hallucinating intermediate classes, which is a known risk if one were to erroneously apply a standard numerical median filter to discrete category indices.

#### 3. Contiguous Merging & Boundary Calculation
Following the smoothing phase, a fusion algorithm iterates through $\hat{Y}$ to merge consecutive identical predictions into contiguous events. If the model predicts an instrument block spanning from index $a$ to index $b$, the algorithm calculates precise chronological boundaries. 

To completely eliminate temporal overlap between adjacent events (a common artifact where the window width $W$ bleeds into the next classification), the onset and offset are tied strictly to the stride timestamps:
* **Onset:** $t_a$
* **Offset:** $t_{b+1}$ (The exact chronological boundary where the model switched its prediction)

#### 4. Heuristic Duration Thresholding
Finally, a duration-based safety threshold is applied. The total duration of a merged event is $\Delta t = t_{b+1} - t_a$. If $\Delta t < 1.0$ seconds, the event is classified as a transient artifact or statistical anomaly and is safely purged from the final annotation log, ensuring only distinct, confident musical performances are recorded.

### Graphical User Interface (GUI) Engineering

To render this complex computational pipeline usable, the system is packaged into a web-based Graphical User Interface (GUI) powered by **Gradio**. This allows human operators to easily drag and drop audio files, visualize the waveform dynamically, and generate downloadable timestamp tables in a completely non-technical, browser-based environment.