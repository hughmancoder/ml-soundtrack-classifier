import os
import sys
import yaml
import torch
import gradio as gr
import pandas as pd

class GUISettings:
    """Configuration settings for the Gradio interface."""
    SERVER_PORT = 7860
    SERVER_NAME = "127.0.0.1"
    SHARE_PUBLIC = False
    
    # Inference Tuning
    STRIDE_SEC = 1.0
    RMS_THRESHOLD = 0.001
    
    # UI Layout
    TABLE_ROW_COUNT = 50

# Add src to python path so we can import our pipeline modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from soundtrack_classifier.cnn.resnet import ResNet18Classifier
from soundtrack_classifier.cnn.inference import run_sliding_window_inference

import subprocess
import tempfile

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def extract_audio(video_path):
    print(f"Extracting audio from {video_path}...")
    temp_dir = tempfile.gettempdir()
    wav_path = os.path.join(temp_dir, "gradio_extracted_audio.wav")
    command = [
        "ffmpeg", "-y", "-i", video_path, "-vn", 
        "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1", wav_path
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return wav_path

def run_gradio_inference(audio_path, model_filename):
    if not audio_path:
        return pd.DataFrame()
        
    video_exts = ('.mp4', '.mkv', '.mov', '.avi', '.m4v', '.mpeg4', '.webm', '.flv')
    if audio_path.lower().endswith(video_exts):
        audio_path = extract_audio(audio_path)
        
    config = load_config()
    if not model_filename:
        model_filename = "resnet18_cnn_best.pt"
    model_path = os.path.join(config['paths']['models_dir'], model_filename)
    
    if not os.path.exists(model_path):
        return pd.DataFrame()
        
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    try:
        print("\n" + "="*50)
        print(f"EVALUATING WITH MODEL: {model_filename}")
        print("="*50 + "\n")
        
        # 1. Load Model
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        labels = config['train_labels']
        state_dict = checkpoint
        
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            if 'classes' in checkpoint:
                labels = checkpoint['classes']
                
        idx_to_class = {i: cls_name for i, cls_name in enumerate(labels)}
        model = ResNet18Classifier(num_classes=len(labels)).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        
        # 2. Run Inference with Post-Processing
        results = run_sliding_window_inference(
            audio_path=audio_path,
            model=model,
            idx_to_class=idx_to_class,
            sr=config['audio']['sample_rate'],
            window_sec=config['audio']['duration_sec'],
            stride_sec=GUISettings.STRIDE_SEC,
            device=device,
            rms_threshold=GUISettings.RMS_THRESHOLD
        )
        
        # 3. Format Output
        def format_time(sec):
            h = int(sec // 3600)
            m = int((sec % 3600) // 60)
            s = int(sec % 60)
            return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"

        df = pd.DataFrame(results, columns=["Start Time", "End Time", "Instrument"])
        
        if not df.empty:
            df["Start Time"] = df["Start Time"].apply(format_time)
            df["End Time"] = df["End Time"].apply(format_time)
            
        return df
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return pd.DataFrame()

def export_csv(df):
    if df is None or df.empty:
        return None
    csv_path = os.path.join(tempfile.gettempdir(), "extracted_events.csv")
    df.to_csv(csv_path, index=False)
    return csv_path

# Define the Gradio Interface
def create_app():
    config = load_config()
    models_dir = config['paths']['models_dir']
    available_models = [f for f in os.listdir(models_dir) if f.endswith('.pt')] if os.path.exists(models_dir) else []
    default_model = "resnet18_cnn_best.pt" if "resnet18_cnn_best.pt" in available_models else (available_models[0] if available_models else None)
    
    with gr.Blocks(title="Soundtrack CNN Classifier") as app:
        gr.Markdown("# ML Soundtrack Classifier")
        
        with gr.Row():
            with gr.Column(scale=1):
                model_dropdown = gr.Dropdown(
                    choices=available_models, 
                    value=default_model,
                    label="Select Model Checkpoint"
                )
                audio_input = gr.File(
                    file_types=["audio", "video", ".mp4", ".m4v", ".mov"], 
                    label="Upload Audio or Video File (.wav, .mp3, .mp4, .m4v, etc.)"
                )
                audio_player = gr.Audio(label="Audio Playback", interactive=False)
                
                with gr.Row():
                    infer_btn = gr.Button("Run Inference", variant="primary", scale=2)
                    export_btn = gr.Button("Save Table to CSV", scale=1)
                    
                csv_download = gr.File(label="⬇ CSV Download", interactive=False, height=70)
                
        with gr.Row():
            results_table = gr.Dataframe(
                headers=["Start Time", "End Time", "Instrument"],
                label="Extracted Acoustic Events",
                interactive=False,
                wrap=True,
                row_count=(GUISettings.TABLE_ROW_COUNT, "dynamic")
            )
            
        def load_audio(file_path):
            if not file_path: return None
            video_exts = ('.mp4', '.mkv', '.mov', '.avi', '.m4v', '.mpeg4', '.webm', '.flv')
            if file_path.lower().endswith(video_exts):
                return extract_audio(file_path)
            return file_path
            
        audio_input.change(
            fn=load_audio,
            inputs=audio_input,
            outputs=audio_player
        )
                
        infer_btn.click(
            fn=run_gradio_inference,
            inputs=[audio_input, model_dropdown],
            outputs=results_table
        )
        
        export_btn.click(
            fn=export_csv,
            inputs=results_table,
            outputs=csv_download
        )
        
    return app

if __name__ == "__main__":
    app = create_app()
    app.launch(
        share=GUISettings.SHARE_PUBLIC, 
        server_name=GUISettings.SERVER_NAME, 
        server_port=GUISettings.SERVER_PORT, 
        theme=gr.themes.Soft()
    )
