import os
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import soundfile as sf


class FilmStemSeparator:
    """
    Phase 5: Cinematic Film Stem Separation (CASS)
    Isolates dialogue (a_DX), sound effects (a_FX), and musical score (a_MX)
    from optical composite monophonic audio using HTDemucs / BandIt.
    """
    def __init__(
        self,
        model_name: str = "htdemucs",
        output_dir: str = "data/stems",
        music_stem_name: str = "other"
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.music_stem_name = music_stem_name

    def separate(
        self,
        audio_path: str,
        force_recompute: bool = False
    ) -> str:
        """
        Executes stem separation on film audio, returning the path to the isolated music stem (a_MX).
        """
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Film audio file not found: {audio_path}")

        track_stem = audio_file.stem
        expected_stem_dir = Path(self.output_dir) / self.model_name / track_stem
        candidate_music = expected_stem_dir / f"{self.music_stem_name}.wav"

        # Check if already separated
        if candidate_music.exists() and not force_recompute:
            print(f"Isolated film score stem already exists at: {candidate_music}")
            return str(candidate_music)

        print(f"Phase 5: Running {self.model_name} separation on {audio_path}...")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        cmd = [
            "demucs",
            "-n", self.model_name,
            "-o", str(self.output_dir),
            str(audio_path)
        ]

        try:
            subprocess.run(cmd, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            # Check if Demucs python module can be run directly
            cmd_py = ["python", "-m", "demucs", "-n", self.model_name, "-o", str(self.output_dir), str(audio_path)]
            subprocess.run(cmd_py, check=True)

        if candidate_music.exists():
            print(f"Separation complete! Isolated music stem (a_MX) saved to: {candidate_music}")
            return str(candidate_music)

        # Look for any wav file in the track directory
        wavs = list(expected_stem_dir.glob("*.wav"))
        for w in wavs:
            if any(name in w.stem.lower() for name in ["other", "music", "score"]):
                return str(w)

        if wavs:
            return str(wavs[0])

        raise FileNotFoundError(f"Could not locate separated music stem in {expected_stem_dir}")
