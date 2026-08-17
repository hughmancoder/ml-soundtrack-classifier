# Packaging & Distribution Guide

This document explains how to package the ML Soundtrack Classifier so that non-technical users can run it offline on macOS.

## For the Developer (How to Package)

To distribute this application, the end-user does **not** need to use `git clone`, set up Docker, or install dependencies manually. You simply need to send them a compressed snapshot of the project.

### Steps to Package:
1. Ensure your `models/` directory contains your best trained weights (e.g., `resnet18_cnn_best.pt`).
2. **Crucial:** Delete the `venv/` folder and any `__pycache__` folders from your project directory. You do not want to send them your 2GB virtual environment; their computer will generate a fresh one automatically!
3. In your macOS Finder, right-click the `ml-soundtrack-classifier` folder and select **Compress "ml-soundtrack-classifier"**.
4. Send the resulting `.zip` file to your user (via Google Drive, Dropbox, USB, etc.).

---

## For the End-User (How to Run)

The application includes a `Start_Mac.command` file. This is a "One-Click Launcher" script designed to completely automate the setup process for a non-technical user without touching code.

### Instructions to give your user:
1. **Unzip** the downloaded folder on your Mac.
2. Open the folder and double-click the **`Start_Mac.command`** file.
3. A Terminal window will open and take over. It will automatically:
   - Check if you have Python 3 installed. *(If you don't, it will safely pause and provide you a direct download link to install it).*
   - Check if you have FFmpeg installed. *(If you don't, it will automatically install it for you).*
   - Create an isolated, hidden Python virtual environment.
   - Download and install all required AI libraries (PyTorch, Gradio, etc.) automatically.
4. Once the automated setup finishes, it will launch the Graphical Interface in your web browser!

> [!NOTE]
> The very first time the user runs this script, it may take 5-10 minutes to download all the PyTorch dependencies (roughly 2GB) depending on their internet speed. Every subsequent launch will be completely offline and take only a few seconds!
