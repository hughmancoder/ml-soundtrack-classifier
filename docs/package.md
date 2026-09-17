# Standalone Application Packaging & Distribution Guide

This document explains how the standalone desktop packaging for the **ML Soundtrack Classifier** works, how to build and update the distribution package, and how to distribute it to non-technical users via Google Drive.

---

## 1. Overview: The Problem & Solution

### The Challenge
Machine learning applications are notoriously difficult to share with non-technical users because they require:
1. Python 3.12+ installed on their machine.
2. Multi-gigabyte ML frameworks (PyTorch, torchaudio, librosa, scipy, etc.).
3. A working `ffmpeg` command-line utility for extracting audio streams from video files (.mp4, .mov, etc.).
4. Command-line literacy (running terminal commands, activating virtual environments).

### The Solution: Native Standalone macOS Bundle
Using **PyInstaller**, the entire Python runtime, PyTorch neural network libraries, Gradio web server, model weights (`models/resnet18_cnn_best.pt`), and a static Apple Silicon `ffmpeg` binary are compiled and bundled into a self-contained macOS `.app` bundle:

```
SoundtrackClassifier_Package/
├── Launch.command                 # Double-clickable launcher (clears Gatekeeper & starts app)
├── README.txt                     # Plaintext 1-minute quickstart guide for the user
└── SoundtrackClassifier.app/      # Native macOS application bundle
    └── Contents/
        ├── MacOS/
        │   ├── SoundtrackClassifier   # Compiled executable
        │   ├── models/                # Trained neural network weights (.pt)
        │   └── config/                # config.yaml settings
        ├── Frameworks/                # PyTorch, Gradio, & dynamic libraries
        └── Resources/                 # Static web assets & templates
```

The user downloads a single `.zip` file from Google Drive, unzips it, double-clicks `Launch.command`, and their web browser automatically opens to the drag-and-drop interface. **No Python installation, no terminal commands, and no internet downloads are required.**

---

## 2. Quick Start: How to Build the Package

To build the standalone macOS package and generate the Google Drive zip file, run a single make command:

```bash
make package-gui
```

### What This Command Does:
1. **Verifies Prerequisites**: Ensures `pyinstaller` and standalone `imageio-ffmpeg` are installed in your virtual environment.
2. **Validates Model Weights**: Checks for `models/resnet18_cnn_best.pt` (or falls back to any valid `.pt` checkpoint in `models/`).
3. **Applies Environment Compatibility Fixes**: Ensures `scipy.stats` and `gradio` dynamic runtime inspections operate smoothly inside frozen binaries.
4. **Executes PyInstaller**: Runs `packaging/soundtrack_classifier.spec` to bundle all binaries and assets into `dist/SoundtrackClassifier.app`.
5. **Assembles Distribution Directory**: Copies the `.app`, embeds `models/` and `config/`, creates the user-facing `Launch.command`, and generates `README.txt`.
6. **Produces Compressed Zip**: Outputs `dist/SoundtrackClassifier-macOS-AppleSilicon.zip` ready to upload to Google Drive.

---

## 3. Architecture & Technical Details

### 3.1 PyInstaller Specification (`packaging/soundtrack_classifier.spec`)

The PyInstaller spec file controls how dependencies are packaged:

* **Static Asset Collection**:
  - `gradio` & `gradio_client`: Bundles HTML/JS/CSS frontend templates and simple dropdown components.
  - `safehttpx` & `groovy`: Bundles internal version metadata files required by Gradio's HTTP server.
  - `imageio_ffmpeg`: Bundles the standalone `ffmpeg` executable.
  - `transformers`, `huggingface_hub`, & `tokenizers`: Bundles tokenizers and model configuration assets.
* **Dynamic Libraries**:
  - Collects PyTorch (`torch`), torchaudio, and NumPy shared C-extensions.
* **Model Checkpoints & Configuration**:
  - Embeds `config/config.yaml` and all `.pt` checkpoints (`models/ast_probing_head.pt`, etc.) directly into the package bundle.
* **Excluded Modules**:
  - Excludes unused, heavy dependencies such as `matplotlib`, `tkinter`, and `wandb` to keep bundle size and startup latency minimal.

### 3.2 Automated Environment Compatibility Fixes

During the packaging script (`scripts/package_mac_app.sh`), several compatibility patches are applied automatically for Python 3.12 and frozen app execution:
1. **`torch._numpy._ufuncs` Global Scope**: Replaces `vars()[name] = ...` with `globals()[name] = ...` to prevent `NameError: name 'name' is not defined` when PyTorch Dynamo and NumPy emulation modules are frozen.
2. **`gradio.component_meta` Type Generation**: Neutralizes `create_or_modify_pyi()` when running inside a frozen app (`sys.frozen`), preventing `FileNotFoundError` caused by missing raw `.py` source text inside the PyInstaller app bundle.
3. **`scipy.stats` Docstring Stripping**: Guards dynamic execution cleanup against missing uninitialized symbols.

### 3.3 Foundation Models & Internet Access Note

* The trained Probing Head (`ast_probing_head.pt`, ~3.2 MB) is bundled directly within the app.
* On the first run, the foundation backbones (**MERT** ~380 MB and **CLAP** ~440 MB) are automatically retrieved and cached locally to `~/.cache/huggingface/`. The user requires an active internet connection on first launch only.

### 3.4 Bundled Static FFmpeg Discovery

In `src/soundtrack_classifier/gui.py`, audio extraction from video files uses `get_ffmpeg_binary()`:
1. First, checks for the bundled static `imageio-ffmpeg` binary (which has zero dynamic dependencies on Homebrew and runs on any Apple Silicon Mac).
2. Second, checks for a bundled binary in the application's `bin/` directory.
3. Third, checks system PATH or standard Homebrew paths (`/opt/homebrew/bin/ffmpeg`).

### 3.5 Auto-Opening Browser on Launch

In `src/soundtrack_classifier/gui.py`, the Gradio launcher specifies `inbrowser=True`. When the user starts the app, Safari or Chrome automatically launches and navigates to:
```
http://127.0.0.1:7860
```

---

## 4. Distributing to the End-User

### Step 1: Upload to Google Drive
1. Locate the generated archive:
   ```
   dist/SoundtrackClassifier-macOS-AppleSilicon.zip
   ```
   *(File size is ~1.1 GB compressed).*
2. Upload this file to your Google Drive folder and copy the shareable download link.

### Step 2: Instructions to Send to the Recipient
Send the following instructions to your user:

> **How to Run the Soundtrack Classifier:**
> 1. Download and unzip the file on your Mac.
> 2. Open the unzipped folder and double-click **`Launch.command`**.
> 3. A small terminal window will open, and your web browser will automatically open the application interface at `http://127.0.0.1:7860`.
> 4. Drag & drop any audio file (.wav, .mp3) or video file (.mp4, .mov), then click **Run Inference**.
> 5. To quit, simply close the small terminal window.
>
> **If macOS shows a Security Warning ("Unidentified Developer"):**
> - Right-click (or Control-click) `Launch.command` and select **Open**.
> - Click **Open** again in the confirmation prompt. You only need to do this once!

---

## 5. Cleaning Caches & Build Artifacts

To clean up build artifacts and caches before or after packaging:
* `make clean`: Cleans all build artifacts (`build/`, `dist/`), Python bytecode, test caches, and AST feature caches.
* `make clean-ast`: Purges AST caches (`data/ast/` standardized audio, segments, features) and separated stems (`data/stems/`).
* `make clean-cache`: Removes `__pycache__`, `.pytest_cache`, `.ruff_cache`, and build directories.

---

## 6. File Structure Reference

| File | Purpose |
| :--- | :--- |
| `Makefile` | Contains `package-gui`, `clean`, `clean-ast`, `clean-cache`, and `clean-data` targets. |
| `scripts/package_mac_app.sh` | Bash script executing assembly, compatibility patching, PyInstaller bundling, and zip creation. |
| `packaging/soundtrack_classifier.spec` | PyInstaller spec defining analysis, collected datas, hidden imports, and bundle settings. |
| `src/soundtrack_classifier/gui.py` | Main Gradio GUI entry point with frozen path handling and auto-browser opening. |
| `dist/SoundtrackClassifier-macOS-AppleSilicon.zip` | Final distributable zip archive for Google Drive. |
