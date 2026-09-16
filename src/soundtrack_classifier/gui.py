import os
import sys
import yaml
import torch
try:
    import gradio.component_meta
    gradio.component_meta.create_or_modify_pyi = lambda *args, **kwargs: None
    import gradio.blocks_events
    gradio.blocks_events.create_or_modify_pyi = lambda *args, **kwargs: None
except Exception:
    pass
import gradio as gr
import pandas as pd

def get_base_dir():
    """Returns the base directory of the project, handling PyInstaller frozen execution."""
    if getattr(sys, 'frozen', False):
        return getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def load_config(config_path="config/config.yaml"):
    base_dir = get_base_dir()
    full_path = os.path.join(base_dir, config_path) if not os.path.isabs(config_path) else config_path
    if not os.path.exists(full_path) and os.path.exists(config_path):
        full_path = config_path
        
    with open(full_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
        
    if 'paths' in config and 'models_dir' in config['paths']:
        if not os.path.isabs(config['paths']['models_dir']):
            config['paths']['models_dir'] = os.path.join(base_dir, config['paths']['models_dir'])
            
    return config

class GUISettings:
    """Configuration settings for the Gradio interface, dynamically backed by config.yaml."""
    _cfg = load_config()
    _crnn_cfg = _cfg.get("crnn", {})
    _crnn_audio = _crnn_cfg.get("audio", {})
    _crnn_inf = _crnn_cfg.get("inference", {})

    SERVER_PORT = 7860
    SERVER_NAME = "127.0.0.1"
    SHARE_PUBLIC = False
    
    # Model Architecture: CRNN exclusively
    DEFAULT_MODEL_TYPE = "CRNN"
    
    # Inference Tuning (CNN - Legacy baseline commented out)
    # STRIDE_SEC = 1.0
    # RMS_THRESHOLD = 0.001
    
    # Inference Tuning (CRNN) - Dynamically loaded from config.yaml
    CRNN_WINDOW_SEC = float(_crnn_audio.get("window_sec", 10.0))
    CRNN_HOP_SEC = float(_crnn_audio.get("hop_sec", 4.0))
    CRNN_BATCH_SIZE = int(_crnn_inf.get("batch_size", 64))
    CRNN_SR = int(_crnn_audio.get("sample_rate", 22050))
    CRNN_RMS_THRESHOLD = float(_crnn_inf.get("rms_threshold", 0.005))
    CRNN_CONFIDENCE_THRESHOLD = float(_crnn_inf.get("confidence_threshold", 0.35))
    CRNN_MIN_EVENT_SEC = float(_crnn_inf.get("min_event_sec", 1.0))
    CRNN_INCLUDE_NONMUSIC = bool(_crnn_inf.get("include_nonmusic", False))
    
    # UI Layout
    TABLE_ROW_COUNT = 50

# Add src to python path so we can import our pipeline modules
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# =============================================================================
# Legacy CNN Baseline imports (Commented out)
# =============================================================================
# from soundtrack_classifier.cnn.resnet import ResNet18Classifier
# from soundtrack_classifier.cnn.inference import run_sliding_window_inference

from soundtrack_classifier.rnn.rnn import ArchivalInstrumentCRNN
from soundtrack_classifier.rnn.inference import run_streaming_rnn_inference

import shutil
import subprocess
import tempfile

def get_ffmpeg_binary():
    """Locate ffmpeg binary in bundled package, imageio_ffmpeg, system path, or homebrew paths."""
    # 1. Check imageio_ffmpeg package for bundled self-contained binary
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        if os.path.exists(ffmpeg_exe) and os.access(ffmpeg_exe, os.X_OK):
            return ffmpeg_exe
    except Exception:
        pass

    base_dir = get_base_dir()
    candidates = [
        os.path.join(base_dir, "bin", "ffmpeg"),
        os.path.join(base_dir, "ffmpeg"),
        os.path.join(os.path.dirname(sys.executable), "bin", "ffmpeg"),
        os.path.join(os.path.dirname(sys.executable), "ffmpeg")
    ]
    for candidate in candidates:
        if os.path.exists(candidate) and os.access(candidate, os.X_OK):
            return candidate
            
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
        
    for mac_path in ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"]:
        if os.path.exists(mac_path) and os.access(mac_path, os.X_OK):
            return mac_path
            
    return "ffmpeg"

def get_available_models(model_type="CRNN"):
    """List available CRNN model checkpoint filenames."""
    config = load_config()
    models_dir = config.get('paths', {}).get('models_dir', 'models')
    if not os.path.isabs(models_dir):
        models_dir = os.path.join(get_base_dir(), models_dir)
        
    if not os.path.exists(models_dir):
        return []
        
    all_models = sorted([f for f in os.listdir(models_dir) if f.endswith('.pt')])
    crnn_models = [f for f in all_models if "crnn" in f.lower() or "rnn" in f.lower()]
    return crnn_models if crnn_models else all_models

    # Legacy CNN model filtering (commented out)
    # if not model_type or "CRNN" in model_type.upper():
    #     crnn_models = [f for f in all_models if "crnn" in f.lower() or "rnn" in f.lower()]
    #     return crnn_models if crnn_models else all_models
    # else:
    #     cnn_models = [f for f in all_models if "crnn" not in f.lower() and ("cnn" in f.lower() or "resnet" in f.lower())]
    #     return cnn_models if cnn_models else all_models

def get_default_model(model_type="CRNN"):
    """Return default CRNN model checkpoint filename."""
    models = get_available_models(model_type)
    if not models:
        return None
    if "crnn_best.pt" in models:
        return "crnn_best.pt"
    if "crnn.pt" in models:
        return "crnn.pt"
    # Legacy CNN default models (commented out)
    # if "resnet18_cnn_best.pt" in models:
    #     return "resnet18_cnn_best.pt"
    # if "resnet18_cnn.pt" in models:
    #     return "resnet18_cnn.pt"
    return models[0]

def extract_audio(video_path):
    print(f"Extracting audio from {video_path}...")
    temp_dir = tempfile.gettempdir()
    wav_path = os.path.join(temp_dir, "gradio_extracted_audio.wav")
    ffmpeg_bin = get_ffmpeg_binary()
    command = [
        ffmpeg_bin, "-y", "-i", video_path, "-vn", 
        "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1", wav_path
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return wav_path

def run_gradio_inference(audio_path, model_filename=None, model_type="CRNN"):
    if not audio_path:
        return pd.DataFrame(columns=["Start Time", "End Time", "Instrument", "Confidence"])
        
    video_exts = ('.mp4', '.mkv', '.mov', '.avi', '.m4v', '.mpeg4', '.webm', '.flv')
    if audio_path.lower().endswith(video_exts):
        audio_path = extract_audio(audio_path)
        
    config = load_config()
    
    if not model_filename:
        model_filename = get_default_model("CRNN")
        if not model_filename:
            print("No CRNN model checkpoint found.")
            return pd.DataFrame(columns=["Start Time", "End Time", "Instrument", "Confidence"])
            
    models_dir = config.get('paths', {}).get('models_dir', 'models')
    model_path = os.path.join(models_dir, model_filename)
    if not os.path.exists(model_path):
        base_model_path = os.path.join(get_base_dir(), 'models', model_filename)
        if os.path.exists(base_model_path):
            model_path = base_model_path
        else:
            print(f"Model file not found: {model_path}")
            return pd.DataFrame(columns=["Start Time", "End Time", "Instrument", "Confidence"])
        
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    try:
        print("\n" + "="*50)
        print(f"EVALUATING WITH CRNN MODEL: {model_filename} on {device}")
        print("="*50 + "\n")
        
        # 1. Load Model Checkpoint & Class Labels
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        labels = config.get('train_labels', [
            "sheng", "dizi", "erhu", "pipa", "suona", "guzheng",
            "yangqin", "percussion", "strings", "woodwind", "brass", "nonmusic"
        ])
        
        if isinstance(checkpoint, dict):
            if 'classes' in checkpoint:
                labels = checkpoint['classes']
            state_dict = checkpoint.get('state_dict', checkpoint)
        else:
            state_dict = checkpoint
                
        idx_to_class = {i: cls_name for i, cls_name in enumerate(labels)}

        def format_time(sec):
            h = int(sec // 3600)
            m = int((sec % 3600) // 60)
            s = int(sec % 60)
            return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"

        # 2. Run CRNN Streaming Inference
        crnn_cfg = config.get("crnn", {})
        crnn_audio = crnn_cfg.get("audio", {})
        crnn_inf = crnn_cfg.get("inference", {})

        sr = int(crnn_audio.get("sample_rate", GUISettings.CRNN_SR))
        window_sec = float(crnn_audio.get("window_sec", GUISettings.CRNN_WINDOW_SEC))
        hop_sec = float(crnn_audio.get("hop_sec", GUISettings.CRNN_HOP_SEC))
        batch_size = int(crnn_inf.get("batch_size", GUISettings.CRNN_BATCH_SIZE))
        rms_threshold = float(crnn_inf.get("rms_threshold", GUISettings.CRNN_RMS_THRESHOLD))
        conf_threshold = float(crnn_inf.get("confidence_threshold", GUISettings.CRNN_CONFIDENCE_THRESHOLD))
        min_event_sec = float(crnn_inf.get("min_event_sec", GUISettings.CRNN_MIN_EVENT_SEC))
        include_nonmusic = bool(crnn_inf.get("include_nonmusic", GUISettings.CRNN_INCLUDE_NONMUSIC))

        model = ArchivalInstrumentCRNN(
            num_classes=len(labels),
            sr=sr
        ).to(device)
        model.load_state_dict(state_dict)
        model.eval()

        events = run_streaming_rnn_inference(
            audio_path=audio_path,
            model=model,
            idx_to_class=idx_to_class,
            sr=sr,
            window_sec=window_sec,
            hop_sec=hop_sec,
            batch_size=batch_size,
            device=device,
            rms_threshold=rms_threshold,
            confidence_threshold=conf_threshold,
            min_event_sec=min_event_sec,
            include_nonmusic=include_nonmusic,
            config=config
        )

        rows = []
        for ev in events:
            rows.append({
                "Start Time": format_time(ev["start_sec"]),
                "End Time": format_time(ev["end_sec"]),
                "Instrument": ev["instrument"],
                "Confidence": f"{ev['confidence']:.2f}"
            })
        return pd.DataFrame(rows, columns=["Start Time", "End Time", "Instrument", "Confidence"])

        # =====================================================================
        # Legacy CNN Baseline Inference (Commented Out)
        # =====================================================================
        # else:
        #     model = ResNet18Classifier(num_classes=len(labels)).to(device)
        #     model.load_state_dict(state_dict)
        #     model.eval()
        #     
        #     results = run_sliding_window_inference(
        #         audio_path=audio_path,
        #         model=model,
        #         idx_to_class=idx_to_class,
        #         sr=config['audio']['sample_rate'],
        #         window_sec=config['audio']['duration_sec'],
        #         stride_sec=GUISettings.STRIDE_SEC,
        #         device=device,
        #         rms_threshold=GUISettings.RMS_THRESHOLD
        #     )
        #     
        #     rows = []
        #     for start_sec, end_sec, class_str in results:
        #         rows.append({
        #             "Start Time": format_time(start_sec),
        #             "End Time": format_time(end_sec),
        #             "Instrument": class_str,
        #             "Confidence": "-"
        #         })
        #     return pd.DataFrame(rows, columns=["Start Time", "End Time", "Instrument", "Confidence"])
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(columns=["Start Time", "End Time", "Instrument", "Confidence"])

def export_csv(df):
    if df is None or df.empty:
        return None
    csv_path = os.path.join(tempfile.gettempdir(), "extracted_events.csv")
    df.to_csv(csv_path, index=False)
    return csv_path

# Define the Gradio Interface
def create_app():
    initial_models = get_available_models("CRNN")
    initial_default = get_default_model("CRNN")
    
    with gr.Blocks(title="ML Soundtrack Classifier (Archival CRNN)") as app:
        gr.Markdown("# ML Soundtrack Classifier (Archival CRNN)")
        gr.Markdown("Identify musical instruments and acoustic events in archival film soundtracks using 2-channel HPSS + CQT streaming CRNN inference.")
        
        with gr.Row():
            with gr.Column(scale=1):
                # =============================================================
                # Legacy Model Architecture Selector (Commented Out)
                # =============================================================
                # model_type_radio = gr.Radio(
                #     choices=["CNN", "CRNN"],
                #     value="CRNN",
                #     label="Model Architecture",
                #     info="Choose between CNN (ResNet-18 baseline) and CRNN (Archival GRU sequence model)"
                # )
                model_dropdown = gr.Dropdown(
                    choices=initial_models, 
                    value=initial_default,
                    label="Select CRNN Model Checkpoint"
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
                headers=["Start Time", "End Time", "Instrument", "Confidence"],
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
            
        # =====================================================================
        # Legacy Architecture Toggle Callback (Commented Out)
        # =====================================================================
        # def on_model_type_change(selected_type):
        #     models = get_available_models(selected_type)
        #     default_m = get_default_model(selected_type)
        #     return gr.Dropdown(choices=models, value=default_m)
        #
        # model_type_radio.change(
        #     fn=on_model_type_change,
        #     inputs=model_type_radio,
        #     outputs=model_dropdown
        # )

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
        theme=gr.themes.Soft(),
        inbrowser=True
    )
