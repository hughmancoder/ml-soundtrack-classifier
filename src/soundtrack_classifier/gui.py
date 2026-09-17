import os
import shutil
import subprocess
import sys
import tempfile

# Prevent Python from shadowing standard library 'ast' module with soundtrack_classifier/ast
_current_dir = os.path.dirname(os.path.abspath(__file__))
if sys.path and os.path.abspath(sys.path[0]) == _current_dir:
    sys.path.pop(0)

_src_dir = os.path.abspath(os.path.join(_current_dir, '..'))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import torch  # noqa: E402
import yaml  # noqa: E402

try:
    import gradio.component_meta
    gradio.component_meta.create_or_modify_pyi = lambda *args, **kwargs: None
    import gradio.blocks_events
    gradio.blocks_events.create_or_modify_pyi = lambda *args, **kwargs: None
except Exception:
    pass
import gradio as gr  # noqa: E402
import pandas as pd  # noqa: E402


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

    _ast_cfg = _cfg.get("ast", {})
    _ast_scoring = _ast_cfg.get("scoring", {})
    _ast_post = _ast_cfg.get("postprocessing", {})
    _ast_sep = _ast_cfg.get("separation", {})

    SERVER_PORT = 7860
    SERVER_NAME = "127.0.0.1"
    SHARE_PUBLIC = False

    # Model Architectures
    MODEL_TYPES = [
        "AST (Foundation Transformer)",
        # "Archival CRNN",
        # "CNN (ResNet-50)"
    ]
    DEFAULT_MODEL_TYPE = "AST (Foundation Transformer)"

    # Inference Tuning (CRNN) - Dynamically loaded from config.yaml
    CRNN_WINDOW_SEC = float(_crnn_audio.get("window_sec", 10.0))
    CRNN_HOP_SEC = float(_crnn_audio.get("hop_sec", 4.0))
    CRNN_BATCH_SIZE = int(_crnn_inf.get("batch_size", 64))
    CRNN_SR = int(_crnn_audio.get("sample_rate", 22050))
    CRNN_RMS_THRESHOLD = float(_crnn_inf.get("rms_threshold", 0.005))
    CRNN_CONFIDENCE_THRESHOLD = float(_crnn_inf.get("confidence_threshold", 0.35))
    CRNN_MIN_EVENT_SEC = float(_crnn_inf.get("min_event_sec", 1.0))
    CRNN_INCLUDE_NONMUSIC = bool(_crnn_inf.get("include_nonmusic", False))

    # Inference Tuning (AST)
    AST_WINDOW_SEC = float(_ast_scoring.get("window_sec", 4.0))
    AST_HOP_SEC = float(_ast_scoring.get("hop_sec", 1.0))
    AST_BATCH_SIZE = int(_ast_scoring.get("batch_size", 32))
    AST_P_HIGH = float(_ast_post.get("p_high", 0.80))
    AST_P_LOW = float(_ast_post.get("p_low", 0.45))
    AST_MEDIAN_FILTER_SEC = float(_ast_post.get("median_filter_sec", 2.0))
    AST_MIN_DURATION_SEC = float(_ast_post.get("min_duration_sec", 1.5))
    AST_SEPARATION_ENABLED = bool(_ast_sep.get("enabled", False))
    AST_ENTROPY_REJECTION = bool(_ast_post.get("entropy_rejection_enabled", True))
    AST_MAX_ENTROPY = float(_ast_post.get("max_normalized_entropy", 0.80))
    AST_MIN_CONFIDENCE = float(_ast_post.get("min_confidence_gate", 0.40))

    # UI Layout
    TABLE_ROW_COUNT = 50


# Model Imports
from soundtrack_classifier.ast.inference import run_ast_inference  # noqa: E402
from soundtrack_classifier.ast.separation import FilmStemSeparator  # noqa: E402

# from soundtrack_classifier.cnn.inference import run_sliding_window_inference
# from soundtrack_classifier.cnn.resnet import ResNet18Classifier, ResNet50Classifier
# from soundtrack_classifier.rnn.inference import run_streaming_rnn_inference
# from soundtrack_classifier.rnn.rnn import ArchivalInstrumentCRNN


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

def get_available_models(model_type="AST (Foundation Transformer)"):
    """List available model checkpoint filenames based on architecture."""
    config = load_config()
    models_dir = config.get('paths', {}).get('models_dir', 'models')
    if not os.path.isabs(models_dir):
        models_dir = os.path.join(get_base_dir(), models_dir)

    if not os.path.exists(models_dir):
        return []

    all_models = sorted([f for f in os.listdir(models_dir) if f.endswith('.pt')])

    # AST foundation model checkpoint selection
    ast_models = [f for f in all_models if "ast" in f.lower() or "probing" in f.lower()]
    return ast_models if ast_models else ["ast_probing_head.pt"]
    # elif "CRNN" in model_type.upper() or "RNN" in model_type.upper():
    #     crnn_models = [f for f in all_models if "crnn" in f.lower() or "rnn" in f.lower()]
    #     return crnn_models if crnn_models else all_models
    # elif "CNN" in model_type.upper() or "RESNET" in model_type.upper():
    #     cnn_models = [f for f in all_models if "crnn" not in f.lower() and ("cnn" in f.lower() or "resnet" in f.lower())]
    #     return cnn_models if cnn_models else all_models
    # return all_models

def get_default_model(model_type="AST (Foundation Transformer)"):
    """Return default model checkpoint filename for architecture."""
    models = get_available_models(model_type)
    if not models:
        return None
    if "ast_probing_head.pt" in models:
        return "ast_probing_head.pt"
    return models[0]
    # elif "CRNN" in model_type.upper() or "RNN" in model_type.upper():
    #     if "crnn_best.pt" in models:
    #         return "crnn_best.pt"
    #     if "crnn.pt" in models:
    #         return "crnn.pt"
    #     return models[0]
    # elif "CNN" in model_type.upper() or "RESNET" in model_type.upper():
    #     if "resnet50_cnn_best.pt" in models:
    #         return "resnet50_cnn_best.pt"
    #     if "resnet50_cnn.pt" in models:
    #         return "resnet50_cnn.pt"
    #     return models[0]
    # return models[0]

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

def run_gradio_inference(
    audio_path,
    model_filename=None,
    model_type="AST (Foundation Transformer)",
    isolate_stems=False,
    p_high=None,
    p_low=None,
    min_duration=None,
    median_filter_sec=None,
    window_sec=None,
    hop_sec=None,
    include_nonmusic=True,
    progress=gr.Progress()
):
    cols = ["Start Time", "End Time", "Duration", "Instrument", "Confidence"]
    empty_df = pd.DataFrame(columns=cols)
    if not audio_path:
        return empty_df, "### Please upload an audio or video file to run inference."

    video_exts = ('.mp4', '.mkv', '.mov', '.avi', '.m4v', '.mpeg4', '.webm', '.flv')
    if audio_path.lower().endswith(video_exts):
        progress(0.02, desc="Extracting audio stream from video container...")
        audio_path = extract_audio(audio_path)

    config = load_config()

    if not model_filename:
        model_filename = get_default_model(model_type)

    models_dir = config.get('paths', {}).get('models_dir', 'models')
    model_path = os.path.join(models_dir, model_filename) if model_filename else ""
    if not os.path.exists(model_path):
        base_model_path = os.path.join(get_base_dir(), 'models', model_filename) if model_filename else ""
        if os.path.exists(base_model_path):
            model_path = base_model_path

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    def format_time(sec):
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"

    # Phase 5: Stem Separation (HTDemucs) if option selected
    if isolate_stems:
        progress(0.08, desc="Phase 5: Isolating musical score stem (HTDemucs)...")
        print(f"\n[Phase 5 Stem Separation] Running HTDemucs on: {audio_path}")
        try:
            separator = FilmStemSeparator(
                model_name=config.get("ast", {}).get("separation", {}).get("model", "htdemucs"),
                output_dir=config.get("ast", {}).get("paths", {}).get("stems_dir", "data/stems")
            )
            separated_stem = separator.separate(audio_path)
            if os.path.exists(separated_stem):
                audio_path = separated_stem
                print(f"[Phase 5 Stem Separation] Clean music stem isolated: {audio_path}")
        except Exception as stem_err:
            print(f"[Phase 5 Stem Separation] Warning: {stem_err}. Proceeding with composite audio.")

    try:
        rows = []
        # =====================================================================
        # 1. AST (Transformer Foundation Model Probing)
        # =====================================================================
        if "AST" in model_type.upper() or "TRANSFORMER" in model_type.upper():
            eff_p_high = float(p_high if p_high is not None else GUISettings.AST_P_HIGH)
            eff_p_low = float(p_low if p_low is not None else GUISettings.AST_P_LOW)
            eff_median = float(median_filter_sec if median_filter_sec is not None else GUISettings.AST_MEDIAN_FILTER_SEC)
            eff_min_dur = float(min_duration if min_duration is not None else GUISettings.AST_MIN_DURATION_SEC)
            eff_window = float(window_sec if window_sec is not None else GUISettings.AST_WINDOW_SEC)
            eff_hop = float(hop_sec if hop_sec is not None else GUISettings.AST_HOP_SEC)

            print("\n" + "="*50)
            print(f"EVALUATING WITH AST (TRANSFORMER): {model_filename} on {device}")
            print(f"Parameters: P_high={eff_p_high}, P_low={eff_p_low}, MedFilter={eff_median}s, MinDur={eff_min_dur}s, Window={eff_window}s, Hop={eff_hop}s")
            print("="*50 + "\n")

            events = run_ast_inference(
                audio_path=audio_path,
                model_path=model_path,
                window_sec=eff_window,
                hop_sec=eff_hop,
                device=device,
                p_high=eff_p_high,
                p_low=eff_p_low,
                median_filter_sec=eff_median,
                min_duration_sec=eff_min_dur,
                entropy_rejection_enabled=GUISettings.AST_ENTROPY_REJECTION,
                max_normalized_entropy=GUISettings.AST_MAX_ENTROPY,
                min_confidence_gate=GUISettings.AST_MIN_CONFIDENCE,
                include_nonmusic=include_nonmusic,
                progress_callback=lambda ratio, msg: progress(ratio, desc=msg),
                config=config,
            )

            for ev in events:
                conf_val = ev["confidence"]
                conf_str = f"{conf_val * 100:.1f}%" if conf_val <= 1.0 else f"{conf_val:.2f}"
                rows.append({
                    "Start Time": format_time(ev["start_time"]),
                    "End Time": format_time(ev["end_time"]),
                    "Duration": f"{ev['duration']:.1f}s",
                    "Instrument": ev["instrument"],
                    "Confidence": conf_str,
                })

        # =====================================================================
        # 2. Archival CRNN (2-Channel HPSS + CQT) [COMMENTED OUT - AST IS DEFAULT]
        # =====================================================================
        # elif "CRNN" in model_type.upper() or "RNN" in model_type.upper():
        #     eff_conf = float(p_high if p_high is not None else GUISettings.CRNN_CONFIDENCE_THRESHOLD)
        #     eff_window = float(window_sec if window_sec is not None else GUISettings.CRNN_WINDOW_SEC)
        #     eff_hop = float(hop_sec if hop_sec is not None else GUISettings.CRNN_HOP_SEC)
        #     eff_min_dur = float(min_duration if min_duration is not None else GUISettings.CRNN_MIN_EVENT_SEC)
        #
        #     print("\n" + "="*50)
        #     print(f"EVALUATING WITH CRNN MODEL: {model_filename} on {device}")
        #     print(f"Parameters: ConfThreshold={eff_conf}, Window={eff_window}s, Hop={eff_hop}s, MinEvent={eff_min_dur}s")
        #     print("="*50 + "\n")
        #
        #     if not os.path.exists(model_path):
        #         print(f"CRNN model file not found: {model_path}")
        #         return empty_df, f"### CRNN model file `{model_filename}` not found on disk."
        #
        #     checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        #     labels = config.get('train_labels', [
        #         "sheng", "dizi", "erhu", "pipa", "suona", "guzheng",
        #         "yangqin", "percussion", "strings", "woodwind", "brass", "nonmusic"
        #     ])
        #
        #     if isinstance(checkpoint, dict):
        #         if 'classes' in checkpoint:
        #             labels = checkpoint['classes']
        #         state_dict = checkpoint.get('state_dict', checkpoint)
        #     else:
        #         state_dict = checkpoint
        #
        #     idx_to_class = {i: cls_name for i, cls_name in enumerate(labels)}
        #
        #     crnn_cfg = config.get("crnn", {})
        #     crnn_audio = crnn_cfg.get("audio", {})
        #     sr = int(crnn_audio.get("sample_rate", GUISettings.CRNN_SR))
        #     batch_size = int(config.get("crnn", {}).get("inference", {}).get("batch_size", GUISettings.CRNN_BATCH_SIZE))
        #     rms_threshold = float(config.get("crnn", {}).get("inference", {}).get("rms_threshold", GUISettings.CRNN_RMS_THRESHOLD))
        #
        #     progress(0.25, desc="Loading CRNN architecture and computing CQT/HPSS features...")
        #     model = ArchivalInstrumentCRNN(
        #         num_classes=len(labels),
        #         sr=sr
        #     ).to(device)
        #     model.load_state_dict(state_dict)
        #     model.eval()
        #
        #     progress(0.50, desc="Streaming CRNN inference across audio buffers...")
        #     events = run_streaming_rnn_inference(
        #         audio_path=audio_path,
        #         model=model,
        #         idx_to_class=idx_to_class,
        #         sr=sr,
        #         window_sec=eff_window,
        #         hop_sec=eff_hop,
        #         batch_size=batch_size,
        #         device=device,
        #         rms_threshold=rms_threshold,
        #         confidence_threshold=eff_conf,
        #         min_event_sec=eff_min_dur,
        #         include_nonmusic=include_nonmusic,
        #         config=config
        #     )
        #
        #     for ev in events:
        #         inst_name = ev["instrument"]
        #         dur = ev["end_sec"] - ev["start_sec"]
        #         conf_val = ev["confidence"]
        #         conf_str = f"{conf_val * 100:.1f}%" if conf_val <= 1.0 else f"{conf_val:.2f}"
        #         rows.append({
        #             "Start Time": format_time(ev["start_sec"]),
        #             "End Time": format_time(ev["end_sec"]),
        #             "Duration": f"{dur:.1f}s",
        #             "Instrument": inst_name,
        #             "Confidence": conf_str
        #         })
        #
        # =====================================================================
        # 3. CNN Baseline (ResNet-50 / ResNet-18) [COMMENTED OUT - AST IS DEFAULT]
        # =====================================================================
        # else:
        #     print("\n" + "="*50)
        #     print(f"EVALUATING WITH CNN MODEL: {model_filename} on {device}")
        #     print(f"Parameters: Window={window_sec}s, Stride={hop_sec}s")
        #     print("="*50 + "\n")
        #
        #     if not os.path.exists(model_path):
        #         print(f"CNN model file not found: {model_path}")
        #         return empty_df, f"### CNN model file `{model_filename}` not found on disk."
        #
        #     checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        #     labels = config.get('train_labels', [
        #         "erhu", "pipa", "dizi", "percussion", "strings_western", "sheng",
        #         "suona", "guzheng", "yangqin", "woodwind", "brass", "nonmusic"
        #     ])
        #     if isinstance(checkpoint, dict):
        #         if 'classes' in checkpoint:
        #             labels = checkpoint['classes']
        #         state_dict = checkpoint.get('state_dict', checkpoint)
        #     else:
        #         state_dict = checkpoint
        #
        #     idx_to_class = {i: c for i, c in enumerate(labels)}
        #     if "50" in (model_filename or "") or (state_dict and any("layer4.2" in k for k in state_dict.keys())):
        #         model = ResNet50Classifier(num_classes=len(labels)).to(device)
        #     else:
        #         model = ResNet18Classifier(num_classes=len(labels)).to(device)
        #
        #     model.load_state_dict(state_dict)
        #     model.eval()
        #
        #     cnn_cfg = config.get('cnn', {})
        #     audio_cfg = cnn_cfg.get('audio', {})
        #     sr = audio_cfg.get('sample_rate', 44100)
        #
        #     eff_window = float(window_sec if window_sec is not None else 4.0)
        #     eff_hop = float(hop_sec if hop_sec is not None else 1.0)
        #     progress(0.40, desc="Running CNN sliding window inference...")
        #     results = run_sliding_window_inference(
        #         audio_path=audio_path,
        #         model=model,
        #         idx_to_class=idx_to_class,
        #         sr=sr,
        #         window_sec=eff_window,
        #         stride_sec=eff_hop,
        #         device=device,
        #         rms_threshold=0.005
        #     )
        #
        #     for start_sec, end_sec, class_str in results:
        #         if not include_nonmusic and class_str.lower() in ["nonmusic", "noise", "silence"]:
        #             continue
        #         dur = end_sec - start_sec
        #         rows.append({
        #             "Start Time": format_time(start_sec),
        #             "End Time": format_time(end_sec),
        #             "Duration": f"{dur:.1f}s",
        #             "Instrument": class_str,
        #             "Confidence": "-"
        #         })

        df = pd.DataFrame(rows, columns=cols)
        progress(1.0, desc="Done!")

        if df.empty:
            summary_md = f"### No acoustic events detected above confidence threshold ({p_high:.2f}).\n*Try lowering the **Detection Threshold** in the Options panel below.*"
        else:
            counts = df["Instrument"].value_counts().to_dict()
            counts_str = ", ".join([f"**{k}** ({v})" for k, v in counts.items()])
            summary_md = f"### Analysis Summary: {len(df)} events detected\n**Active Instruments:** {counts_str}"

        return df, summary_md

    except Exception as e:
        print(f"Error during inference: {e!s}")
        import traceback
        traceback.print_exc()
        return empty_df, f"### Inference Error: `{e!s}`"

def export_csv(df):
    if df is None or df.empty:
        return None
    csv_path = os.path.join(tempfile.gettempdir(), "extracted_events.csv")
    df.to_csv(csv_path, index=False)
    return csv_path

# Define the Gradio Interface
def create_app():
    initial_arch = GUISettings.DEFAULT_MODEL_TYPE
    initial_models = get_available_models(initial_arch)
    initial_default = get_default_model(initial_arch)

    with gr.Blocks(title="ML Soundtrack Classifier") as app:
        gr.Markdown("# ML Soundtrack Classifier")
        # gr.Markdown("Identify musical instruments and acoustic events in archival film soundtracks using **Audio Spectrogram Transformers (MERT + CLAP Foundation Probing)**.")

        with gr.Row():
            with gr.Column(scale=1):
                model_type_radio = gr.Radio(
                    choices=GUISettings.MODEL_TYPES,
                    value=initial_arch,
                    label="Model Architecture",
                    info="Audio Spectrogram Transformer (Foundation Model Probing)"
                )
                model_dropdown = gr.Dropdown(
                    choices=initial_models,
                    value=initial_default,
                    label=f"Select {initial_arch} Checkpoint"
                )
                audio_input = gr.File(
                    file_types=["audio", "video", ".wav", ".mp3", ".flac", ".mp4", ".m4v", ".mov"],
                    label="Upload Audio or Video File (.wav, .mp3, .mp4, .m4v, etc.)"
                )
                audio_player = gr.Audio(label="Audio Playback Preview", interactive=False)

                with gr.Accordion("Inference & Pipeline Options", open=False):
                    isolate_stems_cb = gr.Checkbox(
                        value=GUISettings.AST_SEPARATION_ENABLED,
                        label="Isolate Music Stem with HTDemucs (Phase 5 Stem Separation)",
                        info="Separates dialogue (a_DX) and Foley/SFX (a_FX) from soundtrack music (a_MX) before classification. Recommended for full cinematic film soundtracks."
                    )
                    with gr.Row():
                        p_high_slider = gr.Slider(
                            minimum=0.10, maximum=0.95, value=GUISettings.AST_P_HIGH, step=0.05,
                            label="Detection Threshold (P_high / Confidence)",
                            info="Onset confidence required to trigger an event"
                        )
                        p_low_slider = gr.Slider(
                            minimum=0.05, maximum=0.90, value=GUISettings.AST_P_LOW, step=0.05,
                            label="Release Threshold (P_low - Hysteresis)",
                            info="Offset threshold below which detection turns off"
                        )
                    with gr.Row():
                        min_duration_slider = gr.Slider(
                            minimum=0.1, maximum=5.0, value=GUISettings.AST_MIN_DURATION_SEC, step=0.1,
                            label="Min Event Duration (seconds)",
                            info="Prune false positive spikes shorter than this"
                        )
                        median_filter_slider = gr.Slider(
                            minimum=0.0, maximum=5.0, value=GUISettings.AST_MEDIAN_FILTER_SEC, step=0.25,
                            label="Median Filter Window (seconds)",
                            info="Temporal smoothing window to bridge brief dropouts"
                        )
                    with gr.Row():
                        window_sec_slider = gr.Slider(
                            minimum=1.0, maximum=15.0, value=GUISettings.AST_WINDOW_SEC, step=0.5,
                            label="Analysis Window (seconds)",
                            info="Window length for feature extraction"
                        )
                        hop_sec_slider = gr.Slider(
                            minimum=0.1, maximum=5.0, value=GUISettings.AST_HOP_SEC, step=0.1,
                            label="Hop Size / Stride (seconds)",
                            info="Sliding window temporal resolution"
                        )
                    include_nonmusic_cb = gr.Checkbox(
                        value=True,
                        label="Include Non-Music Events",
                        info="Display speech / Foley / noise events in the results table"
                    )

                with gr.Row():
                    infer_btn = gr.Button("Run Inference", variant="primary", scale=2)
                    export_btn = gr.Button("Save Table to CSV", scale=1)

                csv_download = gr.File(label="CSV Download", interactive=False, height=70)

            with gr.Column(scale=1):
                status_markdown = gr.Markdown("### Ready for Inference\nUpload an audio/video file and click **Run Inference**.")
                results_table = gr.Dataframe(
                    headers=["Start Time", "End Time", "Duration", "Instrument", "Confidence"],
                    label="Extracted Acoustic Events",
                    interactive=False,
                    wrap=True,
                    row_count=(GUISettings.TABLE_ROW_COUNT, "dynamic")
                )

        def load_audio(file_path):
            if not file_path:
                return None
            video_exts = ('.mp4', '.mkv', '.mov', '.avi', '.m4v', '.mpeg4', '.webm', '.flv')
            if file_path.lower().endswith(video_exts):
                return extract_audio(file_path)
            return file_path

        def on_model_type_change(selected_type):
            models = get_available_models(selected_type)
            default_m = get_default_model(selected_type)

            return (
                gr.Dropdown(choices=models, value=default_m, label=f"Select {selected_type} Checkpoint"),
                GUISettings.AST_P_HIGH,
                GUISettings.AST_P_LOW,
                GUISettings.AST_MIN_DURATION_SEC,
                GUISettings.AST_MEDIAN_FILTER_SEC,
                GUISettings.AST_WINDOW_SEC,
                GUISettings.AST_HOP_SEC
            )
            # CRNN and CNN branches commented out - AST is default:
            # elif "CRNN" in selected_type.upper():
            #     return (
            #         gr.Dropdown(choices=models, value=default_m, label=f"Select {selected_type} Checkpoint"),
            #         GUISettings.CRNN_CONFIDENCE_THRESHOLD,
            #         0.20,
            #         GUISettings.CRNN_MIN_EVENT_SEC,
            #         1.0,
            #         GUISettings.CRNN_WINDOW_SEC,
            #         GUISettings.CRNN_HOP_SEC
            #     )
            # else:
            #     return (
            #         gr.Dropdown(choices=models, value=default_m, label=f"Select {selected_type} Checkpoint"),
            #         0.50,
            #         0.30,
            #         1.0,
            #         0.0,
            #         4.0,
            #         1.0
            #     )

        model_type_radio.change(
            fn=on_model_type_change,
            inputs=model_type_radio,
            outputs=[
                model_dropdown,
                p_high_slider,
                p_low_slider,
                min_duration_slider,
                median_filter_slider,
                window_sec_slider,
                hop_sec_slider
            ]
        )

        audio_input.change(
            fn=load_audio,
            inputs=audio_input,
            outputs=audio_player
        )

        infer_btn.click(
            fn=run_gradio_inference,
            inputs=[
                audio_input,
                model_dropdown,
                model_type_radio,
                isolate_stems_cb,
                p_high_slider,
                p_low_slider,
                min_duration_slider,
                median_filter_slider,
                window_sec_slider,
                hop_sec_slider,
                include_nonmusic_cb
            ],
            outputs=[results_table, status_markdown]
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
