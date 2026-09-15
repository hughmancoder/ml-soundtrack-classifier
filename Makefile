.PHONY: train-cnn cnn-pipeline gui summarize-data debug-samples slice-chmusic slice-nonmusic splice-movie clean help test-cnn test_cnn delete-silent

PYTHON ?= $(shell if [ -f "venv/bin/python" ]; then echo "venv/bin/python"; else echo "python3"; fi)

help:
	@echo "Soundtrack Classifier CNN Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make summarize-data       Summarize training data and check for silent files"
	@echo "  make debug-samples        Generate clean vs degraded samples for debugging"
	@echo "  make slice-chmusic        Partition and slice ChMusic raw data into training set"
	@echo "  make slice-nonmusic       Partition and slice movie nonmusic clips"
	@echo "  make train-cnn            Train the ResNet-18 baseline CNN"
	@echo "  make test-cnn             Evaluate CNN model against film cue sheet annotations"
	@echo "  make cnn-pipeline input=...  Run sliding window inference on a film audio file"
	@echo "  make gui                  Launch the desktop GUI app for drag-and-drop inference"
	@echo "  make clean                Clean up Python cache files"

summarize-data:
	@echo "Summarizing training data distribution..."
	PYTHONPATH=src $(PYTHON) scripts/summarize_data.py

debug-samples:
	@echo "Generating clean vs degraded samples in data/sample/..."
	PYTHONPATH=src $(PYTHON) scripts/debug_samples.py

slice-chmusic:
	@echo "Slicing ChMusic dataset into data/train/..."
	PYTHONPATH=src $(PYTHON) scripts/slice_chmusic.py

slice-nonmusic:
	@echo "Slicing nonmusic clips from movie annotations..."
	PYTHONPATH=src $(PYTHON) scripts/slice_nonmusic.py

splice-movie:
	@echo "Splicing film video into 15-minute MP4 clips..."
	PYTHONPATH=src $(PYTHON) scripts/splice_movie.py


train-cnn:
	@echo "Starting CNN training pipeline..."
	PYTHONPATH=src $(PYTHON) src/train_cnn.py

resume-cnn:
	@echo "Resuming CNN training from models/resnet18_cnn.pt..."
	PYTHONPATH=src $(PYTHON) src/train_cnn.py --resume models/resnet18_cnn.pt

audio ?= data/raw/Come Drink With Me.mp4
model ?= models/resnet18_cnn_best.pt
cuesheet ?= data/test/come-drink-with-me/Film Music Cue Sheet for Come Drink with Me (English) - English Version.csv
mapping ?= data/test/come-drink-with-me/instrument_mapping.json
eval_output ?= reports/inference_eval_come_drink_with_me.json

test-cnn:
	@echo "Evaluating CNN model against cue sheet annotations..."
	PYTHONPATH=src $(PYTHON) src/soundtrack_classifier/cnn/test_cnn.py \
		--audio "$(audio)" \
		--model "$(model)" \
		--cuesheet "$(cuesheet)" \
		--mapping "$(mapping)" \
		--output "$(eval_output)"

test_cnn: test-cnn

cnn-pipeline:
	@ifndef input
	$(error input is not set. Usage: make cnn-pipeline input="path/to/audio.wav")
	@endif
	@echo "Starting sliding window inference on $(input)..."
	PYTHONPATH=src $(PYTHON) scripts/cnn_pipeline.py "$(input)"

gui:
	@echo "Launching Desktop GUI application..."
	PYTHONPATH=src $(PYTHON) src/soundtrack_classifier/gui.py

metrics ?= models/training_metrics.csv
output ?= models/training_curves.png

plot-curves:
	@echo "Plotting training curves from $(metrics)..."
	PYTHONPATH=src $(PYTHON) scripts/plot_training_curves.py --metrics "$(metrics)" --output "$(output)"

evaluate-models:
	@echo "Evaluating multiple model checkpoints..."
	PYTHONPATH=src $(PYTHON) scripts/evaluate_models.py

clean:
	@echo "Cleaning up __pycache__ directories..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "Done."

delete-silent:
	@echo "Deleting silent clips..."
	PYTHONPATH=src $(PYTHON) scripts/delete_silent_clips.py
