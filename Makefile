.PHONY: train-rnn resume-rnn rnn-pipeline precompute-features gui package-gui summarize-data debug debug-samples download-archive slice-chmusic slice-film slice-nonmusic splice-movie clean clean-data clean-cache clean-ast help test-rnn test_rnn delete-silent plot-curves evaluate-models train-cnn resume-cnn test-cnn test_cnn cnn-pipeline train-ast ast-pipeline test-ast pytest-ast

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
	@echo "Usage (CNN Baseline - ResNet-50 Transfer Learning):"
	@echo "  make train-cnn               Train ResNet-50 CNN with dynamic random crops"
	@echo "  make resume-cnn              Resume CNN training from models/resnet50_cnn.pt"
	@echo "  make test-cnn                Evaluate CNN against film cue sheet annotations"
	@echo "  make cnn-pipeline input=...  Run sliding-window CNN inference on an audio file"
	@echo ""
	@echo "Data & Debug Utilities:"
	@echo "  make debug                   Generate clean vs degraded samples & CQT plots in data/debug/"
	@echo "  make summarize-data          Summarize training data distribution"
	@echo "  make download-archive        Download recordings from Chinese Music Archive"
	@echo "  make slice-chmusic           Partition and slice ChMusic raw data into training set"
	@echo "  make slice-film              Slice annotated film clips into class folders in data/train/"
	@echo "  make gui                     Launch the desktop GUI app for drag-and-drop inference"
	@echo "  make clean                   Clean all caches, AST features, stems, builds, and bytecode"
	@echo "  make clean-ast               Clean AST cache, features, standardized audio, and stems"
	@echo "  make clean-data              Clean precomputed feature tensors, AST data, stems, and debug files"
	@echo "  make clean-cache             Clean __pycache__, .pytest_cache, .ruff_cache, build, dist, and artifacts"
	@echo ""

summarize-data:
	@echo "Summarizing training data distribution..."
	PYTHONPATH=src $(PYTHON) scripts/data/summarize_data.py

debug:
	@echo "Generating clean vs degraded samples in data/debug/..."
	PYTHONPATH=src $(PYTHON) scripts/data/debug_samples.py --output-dir data/debug --num-samples 3

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
	PYTHONPATH=src $(PYTHON) -m soundtrack_classifier.gui

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

clean-cache:
	@echo "Cleaning up Python cache and build artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.py[cod]" -delete
	find . -type f -name ".DS_Store" -delete
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov build dist *.egg-info
	@echo "Python and build cache cleaned."

clean-ast:
	@echo "Cleaning up AST cache (standardized audio, segments, features, manifests, stems)..."
	rm -rf data/ast
	rm -rf data/stems
	@echo "AST cache cleaned."

clean-data: clean-ast
	@echo "Cleaning up precomputed feature caches and intermediate data..."
	rm -rf data/features_cqt_hpss
	rm -rf data/debug
	@echo "Precomputed feature caches and intermediate data cleaned."

clean: clean-cache clean-data
	@echo "All cached data, AST features, build artifacts, and bytecode successfully cleaned."

delete-silent:
	@echo "Deleting silent clips..."
	PYTHONPATH=src $(PYTHON) scripts/delete_silent_clips.py


# ==============================================================================
# CNN Baseline Commands
# ==============================================================================
cnn_model ?= models/resnet50_cnn.pt
cnn_eval_output ?= reports/cnn_inference_eval_come_drink_with_me.json

train-cnn:
	@echo "Starting CNN training pipeline..."
	PYTHONPATH=src $(PYTHON) src/train_cnn.py

resume-cnn:
	@echo "Resuming CNN training from models/resnet50_cnn.pt..."
	PYTHONPATH=src $(PYTHON) src/train_cnn.py --resume models/resnet50_cnn.pt

test-cnn:
	@echo "Evaluating CNN model against cue sheet annotations..."
	PYTHONPATH=src $(PYTHON) src/soundtrack_classifier/cnn/test_cnn.py \
		--audio "$(audio)" \
		--model "$(cnn_model)" \
		--cuesheet "$(cuesheet)" \
		--mapping "$(mapping)" \
		--output "$(cnn_eval_output)"

test_cnn: test-cnn

cnn-pipeline:
	@if [ -z "$(input)" ]; then \
		echo "Error: input is not set. Usage: make cnn-pipeline input=\"path/to/audio.wav\""; \
		exit 1; \
	fi
	@echo "Starting sliding window inference on $(input)..."
	PYTHONPATH=src $(PYTHON) scripts/cnn_pipeline.py "$(input)"

# ==============================================================================
# AST & Foundation Model Pipeline Commands
# ==============================================================================
ast_model ?= models/ast_probing_head.pt
ast_eval_output ?= reports/ast_inference_eval_come_drink_with_me.json

train-ast:
	@echo "Starting AST Probing Head training pipeline..."
	PYTHONPATH=src $(PYTHON) src/train_ast.py

ast-pipeline:
	@if [ -z "$(input)" ]; then \
		echo "Starting end-to-end AST scoring on default film Come Drink With Me..."; \
		PYTHONPATH=src $(PYTHON) scripts/cli/ast_pipeline.py --phase score; \
	else \
		echo "Starting AST scoring pipeline on $(input)..."; \
		PYTHONPATH=src $(PYTHON) scripts/cli/ast_pipeline.py "$(input)" --phase score; \
	fi

test-ast:
	@echo "Evaluating AST model against cue sheet annotations..."
	PYTHONPATH=src $(PYTHON) src/soundtrack_classifier/ast/test_ast.py \
		--audio "$(audio)" \
		--model "$(ast_model)" \
		--cuesheet "$(cuesheet)" \
		--mapping "$(mapping)" \
		--output "$(ast_eval_output)"

pytest-ast:
	@echo "Running AST test suite..."
	PYTHONPATH=src venv/bin/pytest tests/test_ast.py -v
