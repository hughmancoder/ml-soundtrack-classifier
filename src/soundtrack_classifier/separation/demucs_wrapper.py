import os
import subprocess
from pathlib import Path

class SourceSeparator:
    """
    Wrapper for the Demucs source separation model.
    Uses the 'htdemucs' model by default to unbake the audio mix into multiple stems.
    """
    def __init__(self, model_name: str = "htdemucs"):
        self.model_name = model_name

    def separate_file(self, input_path: str, output_dir: str):
        """
        Runs Demucs source separation on a single audio file.
        
        Args:
            input_path: Path to the pre-processed .wav file.
            output_dir: Directory to save the isolated stems. Demucs will automatically
                        create a subfolder for the model and the track name.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input audio not found: {input_path}")
            
        print(f"Separating {input_path} using {self.model_name}...")
        
        # We use the demucs CLI via subprocess as it gracefully manages 
        # model downloading, memory chunking, and file saving.
        command = [
            "demucs",
            "-n", self.model_name,
            "-o", output_dir,
            input_path
        ]
        
        subprocess.run(command, check=True)
        print(f"Separation complete. Stems saved to {output_dir}")

    def get_stem_paths(self, track_name: str, base_output_dir: str) -> dict:
        """
        Helper to return the paths of the generated stems.
        Demucs outputs to: base_output_dir/model_name/track_name/stem.wav
        """
        track_dir = Path(base_output_dir) / self.model_name / track_name
        
        stems = {
            "vocals": track_dir / "vocals.wav",
            "drums": track_dir / "drums.wav",
            "bass": track_dir / "bass.wav",
            "other": track_dir / "other.wav"
        }
        
        return {name: str(path) for name, path in stems.items() if path.exists()}
