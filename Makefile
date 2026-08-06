.PHONY: train-cnn cnn-pipeline gui summarize-data debug-samples slice-chmusic slice-nonmusic splice-movie clean help

help:
	@echo "Soundtrack Classifier CNN Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make summarize-data       Summarize training data and check for silent files"
	@echo "  make debug-samples        Generate clean vs degraded samples for debugging"
	@echo "  make slice-chmusic        Partition and slice ChMusic raw data into training set"
	@echo "  make slice-nonmusic       Partition and slice movie nonmusic clips"
	@echo "  make train-cnn            Train the ResNet-18 baseline CNN"
	@echo "  make cnn-pipeline input=...  Run sliding window inference on a film audio file"
	@echo "  make gui                  Launch the desktop GUI app for drag-and-drop inference"
	@echo "  make clean                Clean up Python cache files"

summarize-data:
	@echo "Summarizing training data distribution..."
	PYTHONPATH=src python scripts/summarize_data.py

debug-samples:
	@echo "Generating clean vs degraded samples in data/sample/..."
	PYTHONPATH=src python scripts/debug_samples.py

slice-chmusic:
	@echo "Slicing ChMusic dataset into data/train/..."
	PYTHONPATH=src python scripts/slice_chmusic.py

slice-nonmusic:
	@echo "Slicing nonmusic clips from movie annotations..."
	PYTHONPATH=src python scripts/slice_nonmusic.py

splice-movie:
	@echo "Splicing film video into 15-minute MP4 clips..."
	PYTHONPATH=src python scripts/splice_movie.py


train-cnn:
	@echo "Starting CNN training pipeline..."
	PYTHONPATH=src python scripts/train_cnn.py

cnn-pipeline:
	@ifndef input
	$(error input is not set. Usage: make cnn-pipeline input="path/to/audio.wav")
	@endif
	@echo "Starting sliding window inference on $(input)..."
	PYTHONPATH=src python scripts/cnn_pipeline.py "$(input)"

gui:
	@echo "Launching Desktop GUI application..."
	PYTHONPATH=src python scripts/gui.py

clean:
	@echo "Cleaning up __pycache__ directories..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "Done."

make delete-silent:
	@echo "Deleting silent clips..."
	PYTHONPATH=src python scripts/delete_silent_clips.py
