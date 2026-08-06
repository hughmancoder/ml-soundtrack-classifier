# CNN Training Architecture & Future Improvements

## Current State of `train_cnn.py`
Currently, the `train_cnn.py` script serves as a lightweight Minimum Viable Product (MVP) baseline. 

**Does it add synthetic degradation?**
No. It currently trains on the pure, clean audio slices stored in `data/train`. The synthetic degradation pipeline (using `pedalboard` for reverb, compression, bitcrushing, and noise injection) is not yet integrated into the `dataset.py` or `train_cnn.py` scripts.

**Does it output validation metrics or curves?**
No. The MVP script trains on 100% of the dataset and only prints the training loss to the console. It does not split the data into training/validation sets, nor does it log metrics (Accuracy, F1-Score) or generate loss curves.

## Roadmap for Training Improvements

To elevate the training pipeline to production grade, the following features should be implemented in future iterations:

### 1. On-The-Fly Data Degradation
Instead of permanently saving degraded `.wav` files to the hard drive, we can dynamically apply the `pedalboard` degradation pipeline inside the PyTorch `Dataset`.
- **Implementation**: In `dataset.py`, add a stochastic augmentation function. For example, during `__getitem__`, there is a 50% chance to apply `pedalboard` effects (reverb, noise, compression) to the raw audio tensor before the CQT is calculated.
- **Benefit**: This acts as powerful data augmentation, forcing the CNN to generalize across both clean and heavily degraded legacy audio.

### 2. Validation Loop & Metrics
- **Data Splitting**: We need to implement a standard 80/20 train/validation split.
- **Metrics**: After every epoch, the model should be evaluated on the unseen validation set. We will calculate **Validation Loss** and **Accuracy/F1-Score**.
- **Early Stopping**: If the validation loss stops improving, training should halt to prevent overfitting.

### 3. Training Curves (WandB / TensorBoard)
Since `wandb` (Weights & Biases) is already present in your `requirements.txt`, we should integrate it directly into `train_cnn.py`.
- **Tracking**: `wandb.log({"train_loss": loss, "val_loss": val_loss, "val_acc": accuracy})`
- **Visualization**: This will automatically generate beautiful, real-time training and validation curves in your browser, along with tracking system metrics (GPU utilization, memory).
