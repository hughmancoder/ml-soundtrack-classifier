.PHONY: train-cnn resume-cnn cnn-pipeline train-rnn resume-rnn rnn-pipeline gui package-gui summarize-data debug-samples download-archive slice-chmusic slice-nonmusic splice-movie clean help test-cnn test_cnn test-rnn test_rnn delete-silent

PYTHON ?= $(shell if [ -f "venv/bin/python" ]; then echo "venv/bin/python"; else echo "python3"; fi)

help:
	@echo "Soundtrack Classifier Makefile"
	@echo ""
	@echo "Usage (CNN Baseline):"
	@echo "  make train-cnn               Train the ResNet-18 baseline CNN"
	@echo "  make resume-cnn              Resume CNN training from models/resnet18_cnn.pt"
	@echo "  make test-cnn                Evaluate CNN model against film cue sheet annotations"
	@echo "  make cnn-pipeline input=...  Run sliding window inference on a film audio file"
	@echo ""
	@echo "Usage (Archival CRNN - 6-10s Window & 2-Channel HPSS+CQT):"
	@echo "  make train-rnn               Train the Archival CRNN with dynamic 6-8s crops"
	@echo "  make resume-rnn              Resume CRNN training from models/crnn.pt"
	@echo "  make test-rnn                Evaluate CRNN model against film cue sheet annotations"
	@echo "  make rnn-pipeline input=...  Run streaming 10s Hanning-overlap inference on an audio file"
	@echo ""
	@echo "Data & Utilities:"
	@echo "  make summarize-data          Summarize training data and check for silent files"
	@echo "  make debug-samples           Generate clean vs degraded samples for debugging"
	@echo "  make download-archive        Download recordings from Chinese Music Archive"
	@echo "  make slice-chmusic           Partition and slice ChMusic raw data into training set"
	@echo "  make slice-nonmusic          Partition and slice movie nonmusic clips"
	@echo "  make gui                     Launch the desktop GUI app for drag-and-drop inference"
	@echo "  make clean                   Clean up Python cache files"

summarize-data:
	@echo "Summarizing training data distribution..."
	PYTHONPATH=src $(PYTHON) scripts/summarize_data.py

debug-samples:
	@echo "Generating clean vs degraded samples in data/sample/..."
	PYTHONPATH=src $(PYTHON) scripts/debug_samples.py

download-archive:
	@echo "Downloading Chinese Music Archive recordings..."
	PYTHONPATH=src $(PYTHON) scripts/download_ch_music_archive.py

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
model ?= models/resnet18_cnn_80_percent_degraded.pt
# models/resnet18_cnn_best.pt
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
	@if [ -z "$(input)" ]; then \
		echo "Error: input is not set. Usage: make cnn-pipeline input=\"path/to/audio.wav\""; \
		exit 1; \
	fi
	@echo "Starting sliding window inference on $(input)..."
	PYTHONPATH=src $(PYTHON) scripts/cnn_pipeline.py "$(input)"


train-rnn:
	@echo "Starting Archival CRNN training pipeline with dynamic 6-8s crops..."
	PYTHONPATH=src $(PYTHON) src/train_rnn.py

resume-rnn:
	@echo "Resuming CRNN training from models/crnn.pt..."
	PYTHONPATH=src $(PYTHON) src/train_rnn.py --resume models/crnn.pt

rnn_model ?= models/crnn_best.pt
rnn_eval_output ?= reports/inference_eval_crnn_come_drink_with_me.json

test-rnn:
	@echo "Evaluating CRNN model against cue sheet annotations..."
	PYTHONPATH=src $(PYTHON) src/soundtrack_classifier/rnn/test_rnn.py \
		--audio "$(audio)" \
		--model "$(rnn_model)" \
		--cuesheet "$(cuesheet)" \
		--mapping "$(mapping)" \
		--output "$(rnn_eval_output)"



rnn-pipeline:
	@if [ -z "$(input)" ]; then \
		echo "Error: input is not set. Usage: make rnn-pipeline input=\"path/to/audio.wav\""; \
		exit 1; \
	fi
	@echo "Starting streaming 10s Hanning-overlap inference on $(input)..."
	PYTHONPATH=src $(PYTHON) scripts/rnn_pipeline.py "$(input)"

gui:
	@echo "Launching Desktop GUI application..."
	PYTHONPATH=src $(PYTHON) src/soundtrack_classifier/gui.py

package-gui:
	@echo "Packaging standalone macOS GUI application and zip for distribution..."
	bash scripts/package_mac_app.sh

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
