import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot_curves(metrics_csv, output_image):
    if not os.path.exists(metrics_csv):
        print(f"Error: Could not find metrics file at {metrics_csv}")
        print("Make sure you have run training first!")
        return

    # Load metrics
    df = pd.read_csv(metrics_csv)

    if df.empty:
        print("Metrics file is empty. No data to plot.")
        return

    # Create figure with 2 subplots (Loss and Accuracy)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Loss
    ax1.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o', color='blue')
    ax1.plot(df['epoch'], df['val_loss'], label='Val Loss', marker='o', color='orange')
    ax1.set_title('Training vs Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Plot 2: Accuracy
    ax2.plot(df['epoch'], df['val_acc'], label='Val Accuracy', marker='o', color='green')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.suptitle("CNN Training Curves", fontsize=16)
    plt.tight_layout()

    # Save to file
    plt.savefig(output_image, dpi=300)
    print(f"Successfully generated training curves at: {output_image}")
    
    # Also show the plot if running interactively
    try:
        plt.show(block=False)
        plt.pause(3)
        plt.close()
    except Exception:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training curves from metrics CSV")
    parser.add_argument("--metrics", type=str, default="models/training_metrics.csv", help="Path to the metrics CSV file")
    parser.add_argument("--output", type=str, default="models/training_curves.png", help="Path to save the output PNG plot")
    
    args = parser.parse_args()
    plot_curves(args.metrics, args.output)
