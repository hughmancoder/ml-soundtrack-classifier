.PHONY: train-rnn resume-rnn rnn-pipeline precompute-features gui package-gui summarize-data debug debug-samples download-archive slice-chmusic slice-film slice-nonmusic splice-movie clean help test-rnn test_rnn delete-silent plot-curves evaluate-models

PYTHON ?= $(shell if [ -f "venv/bin/python" ]; then echo "venv/bin/python"; else echo "python3"; fi)

help:
	@echo "Soundtrack Classifier Makefile"
	@echo ""
	@echo "Usage (Archival CRNN - 6-10s Window & 2-Channel HPSS+CQT):"
	@echo "  make train-rnn               Train the Archival CRNN with dynamic 6-8s crops"
	@echo "  make resume-rnn              Resume CRNN training from models/crnn.pt"
	@echo "  make test-rnn                Evaluate CRNN model against film cue sheet annotations"
	@echo "  make rnn-pipeline input=...  Run streaming 10s Hanning-overlap inference on an audio file"
	@echo "  make precompute-features     Pre-compute and cache 2-channel CQT+HPSS tensors for 5-10x faster training"
	@echo ""
	@echo "Data & Debug Utilities:"
	@echo "  make debug                   Generate clean vs degraded samples & CQT plots in data/debug/"
	@echo "  make summarize-data          Summarize training data distribution"
	@echo "  make download-archive        Download recordings from Chinese Music Archive"
	@echo "  make slice-chmusic           Partition and slice ChMusic raw data into training set"
	@echo "  make slice-film              Slice annotated film clips into class folders in data/train/"
	@echo "  make gui                     Launch the desktop GUI app for drag-and-drop inference"
	@echo "  make clean                   Clean up Python cache files"
	@echo ""

summarize-data:
	@echo "Summarizing training data distribution..."
	PYTHONPATH=src $(PYTHON) scripts/summarize_data.py

debug:
	@echo "Generating clean vs degraded samples in data/debug/..."
	PYTHONPATH=src $(PYTHON) scripts/debug_samples.py --output-dir data/debug --num-samples 3

debug-samples: debug

download-archive:
	@echo "Downloading Chinese Music Archive recordings..."
	PYTHONPATH=src $(PYTHON) scripts/download_ch_music_archive.py

slice-chmusic:
	@echo "Slicing ChMusic dataset into data/train/..."
	PYTHONPATH=src $(PYTHON) scripts/data/slice_chmusic.py

slice-film:
	@echo "Slicing annotated film clips into class folders in data/train/..."
	PYTHONPATH=src $(PYTHON) scripts/data/slice_film.py

slice-nonmusic: slice-film

splice-movie:
	@echo "Splicing film video into 15-minute MP4 clips..."
	PYTHONPATH=src $(PYTHON) scripts/splice_movie.py

train-rnn:
	@echo "Starting Archival CRNN training pipeline with dynamic 6-8s crops..."
	PYTHONPATH=src $(PYTHON) src/train_rnn.py

resume-rnn:
	@echo "Resuming CRNN training from models/crnn.pt..."
	PYTHONPATH=src $(PYTHON) src/train_rnn.py --resume models/crnn.pt

audio ?= data/test/come-drink-with-me/Come Drink With Me.wav
cuesheet ?= data/test/come-drink-with-me/Film Music Cue Sheet for Come Drink with Me (English) - English Version.csv
mapping ?= data/test/come-drink-with-me/instrument_mapping.json
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

test_rnn: test-rnn

precompute-features:
	@echo "Precomputing and caching 2-channel CQT+HPSS feature tensors..."
	PYTHONPATH=src $(PYTHON) scripts/precompute_cqt_hpss.py

rnn-pipeline:
	@if [ -z "$(input)" ]; then \
		echo "Error: input is not set. Usage: make rnn-pipeline input=\"path/to/audio.wav\""; \
		exit 1; \
	fi
	@echo "Starting streaming 10s Hanning-overlap inference on $(input)..."
	PYTHONPATH=src $(PYTHON) scripts/cli/rnn_pipeline.py "$(input)"

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


# ==============================================================================
# Legacy CNN Baseline Commands (Commented Out)
# ==============================================================================
# cnn_model ?= models/resnet18_cnn_80_percent_degraded.pt
# cnn_eval_output ?= reports/inference_eval_come_drink_with_me.json
# 
# train-cnn:
# 	@echo "Starting CNN training pipeline..."
# 	PYTHONPATH=src $(PYTHON) src/train_cnn.py
# 
# resume-cnn:
# 	@echo "Resuming CNN training from models/resnet18_cnn.pt..."
# 	PYTHONPATH=src $(PYTHON) src/train_cnn.py --resume models/resnet18_cnn.pt
# 
# test-cnn:
# 	@echo "Evaluating CNN model against cue sheet annotations..."
# 	PYTHONPATH=src $(PYTHON) src/soundtrack_classifier/cnn/test_cnn.py \
# 		--audio "$(audio)" \
# 		--model "$(cnn_model)" \
# 		--cuesheet "$(cuesheet)" \
# 		--mapping "$(mapping)" \
# 		--output "$(cnn_eval_output)"
# 
# test_cnn: test-cnn
# 
# cnn-pipeline:
# 	@if [ -z "$(input)" ]; then \
# 		echo "Error: input is not set. Usage: make cnn-pipeline input=\"path/to/audio.wav\""; \
# 		exit 1; \
# 	fi
# 	@echo "Starting sliding window inference on $(input)..."
# 	PYTHONPATH=src $(PYTHON) scripts/cnn_pipeline.py "$(input)"
