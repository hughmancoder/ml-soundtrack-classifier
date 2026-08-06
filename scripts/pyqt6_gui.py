import sys
import os
import yaml
import torch
import librosa
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
import pyqtgraph as pg

# Add src to python path dynamically so we don't have to rely on PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from soundtrack_classifier.cnn.resnet import ResNet18Classifier
from soundtrack_classifier.cnn.inference import run_sliding_window_inference

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class InferenceWorker(QThread):
    finished = pyqtSignal(list)
    error = pyqtSignal(str)
    
    def __init__(self, audio_path, model_path, config):
        super().__init__()
        self.audio_path = audio_path
        self.model_path = model_path
        self.config = config
        
    def run(self):
        try:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            
            # Load model
            checkpoint = torch.load(self.model_path, map_location=device, weights_only=False)
            labels = self.config['train_labels']
            state_dict = checkpoint
            
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                if 'classes' in checkpoint:
                    labels = checkpoint['classes']
                    
            idx_to_class = {i: cls_name for i, cls_name in enumerate(labels)}
            model = ResNet18Classifier(num_classes=len(labels)).to(device)
            model.load_state_dict(state_dict)
            model.eval()
            
            # Run inference
            results = run_sliding_window_inference(
                audio_path=self.audio_path,
                model=model,
                idx_to_class=idx_to_class,
                sr=self.config['audio']['sample_rate'],
                window_sec=self.config['audio']['duration_sec'],
                stride_sec=1.0,
                device=device,
                rms_threshold=0.005
            )
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))

class SoundtrackGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Soundtrack CNN Classifier")
        self.resize(1000, 700)
        self.setAcceptDrops(True)
        
        self.config = load_config()
        self.current_audio_path = None
        
        # UI Setup
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Drop Zone / Status
        self.status_label = QLabel("Drag & Drop an Audio File Here (.wav, .mp3, .mp4)\nOr click 'Load Audio'")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("QLabel { background-color : #2e2e2e; color : white; border: 2px dashed #666; font-size: 16px; padding: 20px;}")
        layout.addWidget(self.status_label)
        
        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_load = QPushButton("Load Audio")
        self.btn_load.clicked.connect(self.load_audio_dialog)
        self.btn_infer = QPushButton("Run Inference")
        self.btn_infer.clicked.connect(self.start_inference)
        self.btn_infer.setEnabled(False)
        
        btn_layout.addWidget(self.btn_load)
        btn_layout.addWidget(self.btn_infer)
        layout.addLayout(btn_layout)
        
        # Waveform Plot (pyqtgraph)
        self.plot_widget = pg.PlotWidget(title="Audio Waveform")
        self.plot_widget.setYRange(-1, 1)
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_curve = self.plot_widget.plot(pen='y')
        layout.addWidget(self.plot_widget)
        
        # Results Table
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Start Time (s)", "End Time (s)", "Instrument"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)
        
        self.worker = None

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            self.load_audio_file(file_path)

    def load_audio_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Audio File", "", "Audio Files (*.wav *.mp3 *.mp4)")
        if file_path:
            self.load_audio_file(file_path)
            
    def load_audio_file(self, file_path):
        self.current_audio_path = file_path
        self.status_label.setText(f"Loaded: {os.path.basename(file_path)}")
        self.btn_infer.setEnabled(True)
        
        # Load and plot waveform in UI thread (librosa load might block slightly but ok for UI load step)
        try:
            # We downsample significantly just for the UI waveform plot to keep it snappy
            y, sr = librosa.load(file_path, sr=8000, mono=True)
            times = np.linspace(0, len(y)/sr, num=len(y))
            self.plot_curve.setData(times, y)
            self.plot_widget.setXRange(0, times[-1])
        except Exception as e:
            self.status_label.setText(f"Error loading audio: {e}")

    def start_inference(self):
        model_path = os.path.join(self.config['paths']['models_dir'], "resnet18_cnn.pt")
        if not os.path.exists(model_path):
            self.status_label.setText(f"Model not found at {model_path}. Train first!")
            return
            
        self.btn_infer.setEnabled(False)
        self.status_label.setText("Running inference... Please wait (this may take a minute).")
        self.table.setRowCount(0)
        
        self.worker = InferenceWorker(self.current_audio_path, model_path, self.config)
        self.worker.finished.connect(self.on_inference_finished)
        self.worker.error.connect(self.on_inference_error)
        self.worker.start()
        
    def on_inference_finished(self, results):
        self.status_label.setText(f"Inference Complete! Extracted {len(results)} events.")
        self.btn_infer.setEnabled(True)
        
        self.table.setRowCount(len(results))
        for row, (start_t, end_t, label) in enumerate(results):
            self.table.setItem(row, 0, QTableWidgetItem(f"{start_t:.1f}"))
            self.table.setItem(row, 1, QTableWidgetItem(f"{end_t:.1f}"))
            self.table.setItem(row, 2, QTableWidgetItem(label))

    def on_inference_error(self, err_msg):
        self.status_label.setText(f"Inference Error: {err_msg}")
        self.btn_infer.setEnabled(True)

def main():
    app = QApplication(sys.argv)
    
    # Simple dark theme
    app.setStyle("Fusion")
    
    window = SoundtrackGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
