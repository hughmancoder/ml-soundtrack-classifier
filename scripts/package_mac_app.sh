#!/usr/bin/env bash
set -e

# ==============================================================================
# Soundtrack Classifier - macOS Standalone App Packager
# ==============================================================================

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

echo "=========================================================="
echo " Packaging Soundtrack Classifier GUI for macOS (Apple Silicon)"
echo "=========================================================="

# Ensure pyinstaller and imageio-ffmpeg are available
if [ -f "venv/bin/pyinstaller" ]; then
    PYINSTALLER="venv/bin/pyinstaller"
else
    echo "Installing PyInstaller..."
    venv/bin/python -m pip install pyinstaller
    PYINSTALLER="venv/bin/pyinstaller"
fi

# Ensure model weights exist
MODEL_PATH="models/ast_probing_head.pt"
if [ ! -f "$MODEL_PATH" ]; then
    echo "Warning: $MODEL_PATH not found! Falling back to any available .pt model in models/"
    MODEL_PATH=$(ls -1 models/*.pt 2>/dev/null | head -n 1)
    if [ -z "$MODEL_PATH" ]; then
        echo "Error: No model checkpoint found in models/. Train a model first before packaging."
        exit 1
    fi
fi
echo "Using model checkpoint: $MODEL_PATH"

# Mitigate known PyInstaller docstring stripping bug in scipy.stats
python3 -c "
import os
p = 'venv/lib/python3.12/site-packages/scipy/stats/_distn_infrastructure.py'
if os.path.exists(p):
    with open(p, 'r') as f:
        c = f.read()
    target = 'exec(\x27del \x27 + obj)\ndel obj'
    repl = 'try:\n        exec(\x27del \x27 + obj)\n    except Exception:\n        pass\ntry:\n    del obj\nexcept Exception:\n    pass'
    if target in c:
        with open(p, 'w') as f:
            f.write(c.replace(target, repl))
"

# Mitigate known PyInstaller NameError bug in torch._numpy._ufuncs on Python 3.12
python3 -c "
import os
p = 'venv/lib/python3.12/site-packages/torch/_numpy/_ufuncs.py'
if os.path.exists(p):
    with open(p, 'r') as f:
        c = f.read()
    c = c.replace('vars()[name] = deco_binary_ufunc(ufunc)', 'globals()[name] = deco_binary_ufunc(ufunc)')
    c = c.replace('vars()[name] = deco_unary_ufunc(ufunc)', 'globals()[name] = deco_unary_ufunc(ufunc)')
    with open(p, 'w') as f:
        f.write(c)
"

# Mitigate known PyInstaller FileNotFoundError in gradio.component_meta.create_or_modify_pyi
python3 -c "
import os
p = 'venv/lib/python3.12/site-packages/gradio/component_meta.py'
if os.path.exists(p):
    with open(p, 'r') as f:
        c = f.read()
    if 'if getattr(sys, \"frozen\", False):' not in c:
        target = 'def create_or_modify_pyi(\n    component_class: type, class_name: str, events: list[str | EventListener]\n):'
        repl = target + '\n    import sys\n    if getattr(sys, \"frozen\", False):\n        return'
        c = c.replace(target, repl)
        c = c.replace('source_code = source_file.read_text(encoding=\"utf-8\")', 'if not source_file.is_file():\n        return\n    try:\n        source_code = source_file.read_text(encoding=\"utf-8\")\n    except Exception:\n        return')
        with open(p, 'w') as f:
            f.write(c)
"

# Run PyInstaller build
echo "Running PyInstaller..."
$PYINSTALLER --clean --noconfirm packaging/soundtrack_classifier.spec

DIST_DIR="dist/SoundtrackClassifier_Package"
rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR"

echo "Assembling distribution directory at $DIST_DIR..."

# Copy the generated .app into the distribution package
cp -R dist/SoundtrackClassifier.app "$DIST_DIR/"

# Ensure models directory and config are included in both .app and package
mkdir -p "$DIST_DIR/SoundtrackClassifier.app/Contents/MacOS/models"
mkdir -p "$DIST_DIR/SoundtrackClassifier.app/Contents/MacOS/config"
cp models/*.pt "$DIST_DIR/SoundtrackClassifier.app/Contents/MacOS/models/" 2>/dev/null || true
cp config/config.yaml "$DIST_DIR/SoundtrackClassifier.app/Contents/MacOS/config/"

# Copy gradio source files into Contents/Frameworks/gradio to satisfy dynamic inspections
mkdir -p "$DIST_DIR/SoundtrackClassifier.app/Contents/Frameworks/gradio"
cp -R venv/lib/python3.12/site-packages/gradio/ "$DIST_DIR/SoundtrackClassifier.app/Contents/Frameworks/gradio/"

# Create double-clickable Launch.command
cat << 'EOF' > "$DIST_DIR/Launch.command"
#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo "====================================================="
echo " Starting Soundtrack Classifier GUI..."
echo "====================================================="
echo "Note: The web interface will open automatically in your browser."
echo "Keep this window open while using the application."
echo "To exit, close this window or press Ctrl+C."
echo "-----------------------------------------------------"

# Clear macOS quarantine flags from downloaded zip
xattr -cr "$DIR/SoundtrackClassifier.app" 2>/dev/null || true

# Execute the application binary
"$DIR/SoundtrackClassifier.app/Contents/MacOS/SoundtrackClassifier"
EOF

chmod +x "$DIST_DIR/Launch.command"

# Create README.txt for non-technical user
cat << 'EOF' > "$DIST_DIR/README.txt"
=====================================================
 SOUNDTRACK CLASSIFIER - QUICK START GUIDE
=====================================================

Thank you for downloading the Soundtrack Classifier!

HOW TO RUN:
1. Double-click "Launch.command" in this folder.
2. A small terminal window will open, and your default web browser 
   (Safari or Chrome) will automatically open the application interface 
   at: http://127.0.0.1:7860
3. Drag & drop any audio file (.wav, .mp3) or video file (.mp4, .mov, etc.).
4. Click "Run Inference". The detected instruments and soundtrack 
   events will be displayed in the results table.
5. Click "Save Table to CSV" to export your results.

WHEN YOU ARE FINISHED:
Simply close the small terminal window to stop the application.

TROUBLESHOOTING (macOS Security Warning):
If macOS displays a message saying the application "cannot be opened because 
it is from an unidentified developer":
- Right-click (or Control-click) "Launch.command", then click "Open".
- Click "Open" again in the security prompt.
This only needs to be done once.
=====================================================
EOF

# Create Zip Archive for Google Drive
ZIP_NAME="SoundtrackClassifier-macOS-AppleSilicon.zip"
echo "Creating zip archive: dist/$ZIP_NAME ..."
cd dist
rm -f "$ZIP_NAME"
zip -r -q "$ZIP_NAME" "SoundtrackClassifier_Package"
cd "$REPO_DIR"

ZIP_SIZE=$(ls -lh "dist/$ZIP_NAME" | awk '{print $5}')

echo "=========================================================="
echo " Packaging Complete!"
echo " Output Zip: dist/$ZIP_NAME ($ZIP_SIZE)"
echo " You can now upload this zip file directly to Google Drive!"
echo "=========================================================="
