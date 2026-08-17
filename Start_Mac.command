#!/bin/bash

# Move to the directory where this script is located
cd "$(dirname "$0")"

echo "=========================================="
echo " Starting ML Soundtrack Classifier..."
echo "=========================================="

# Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed!"
    echo "Please download and install Python from https://www.python.org/downloads/mac-osx/"
    echo "Press any key to exit..."
    read -n 1
    exit 1
fi

# Check for FFmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "[WARNING] FFmpeg is not installed. It is required for video extraction."
    if command -v brew &> /dev/null; then
        echo "Homebrew detected. Installing FFmpeg..."
        brew install ffmpeg
    else
        echo "[ERROR] Please install FFmpeg or Homebrew first."
        echo "To install Homebrew, open Terminal and paste:"
        echo '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
        echo "Then run this script again."
        echo "Press any key to exit..."
        read -n 1
        exit 1
    fi
fi

# Set up Virtual Environment
if [ ! -d "venv" ]; then
    echo "First time setup: Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements if they aren't installed yet
# We check if gradio is installed as a proxy for the env being ready
if ! python3 -c "import gradio" &> /dev/null; then
    echo "Installing required AI libraries (this may take a few minutes on the first run)..."
    pip install --upgrade pip
    pip install -r requirements.txt
fi

echo "All dependencies verified!"
echo "Launching the Graphical User Interface..."
echo "Please wait a moment. The interface will open in your browser automatically."

# Run the Gradio app
export PYTHONPATH=src
python3 src/soundtrack_classifier/gui.py

echo "Closed successfully."
