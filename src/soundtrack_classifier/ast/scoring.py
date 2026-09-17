from pathlib import Path
from typing import List, Optional, Tuple

import librosa
import numpy as np
import torch
from tqdm import tqdm

from soundtrack_classifier.ast.feature_extraction import FrozenFeatureExtractor
from soundtrack_classifier.ast.probing_head import ProbingHead


class ContinuousFilmScorer:
    """
    Phase 6: Continuous Film Scoring
    Streams 4.0s sliding windows with 0.5s hop across the isolated film score (a_MX).
    Extracts frozen MERT+CLAP embeddings and passes them through the trained probing head.
    Outputs raw probability matrix P in R^{T x 12} across the film timeline.
    """
    def __init__(
        self,
        probing_head_checkpoint: str = "models/ast_probing_head.pt",
        window_sec: float = 4.0,
        hop_sec: float = 0.5,
        sr_mert: int = 24000,
        sr_clap: int = 48000,
        batch_size: int = 32,
        device: Optional[str] = None
    ):
        self.probing_head_checkpoint = probing_head_checkpoint
        self.window_sec = window_sec
        self.hop_sec = hop_sec
        self.sr_mert = sr_mert
        self.sr_clap = sr_clap
        self.batch_size = batch_size

        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.feature_extractor = FrozenFeatureExtractor(
            device=str(self.device),
            batch_size=batch_size
        )
        self.probing_head: Optional[ProbingHead] = None

    def load_probing_head(self):
        """
        Loads the trained probing head checkpoint.
        """
        if self.probing_head is not None:
            return

        chk_path = Path(self.probing_head_checkpoint)
        if not chk_path.exists():
            raise FileNotFoundError(f"Probing head checkpoint not found: {chk_path}")

        checkpoint = torch.load(str(chk_path), map_location="cpu")
        in_dim = checkpoint.get("in_dim", 1280)
        hidden_dims = checkpoint.get("hidden_dims", [512, 256])
        num_classes = checkpoint.get("num_classes", 12)
        dropouts = checkpoint.get("dropouts", [0.4, 0.3])

        self.probing_head = ProbingHead(
            in_dim=in_dim,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            dropouts=dropouts
        )
        self.probing_head.load_state_dict(checkpoint["state_dict"])
        self.probing_head.to(self.device)
        self.probing_head.eval()
        print(f"Loaded trained ProbingHead from {chk_path}")

    def score_audio_file(
        self,
        audio_path: str,
        output_probabilities_path: Optional[str] = "reports/ast_probing_raw_probabilities.npy"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Processes entire audio file in sliding windows and returns:
            (probabilities_matrix, timestamps_sec)
        where probabilities_matrix has shape [T, num_classes].
        """
        self.load_probing_head()
        print(f"Phase 6: Continuous scoring on {audio_path}...")
        print(f"Window: {self.window_sec}s, Hop: {self.hop_sec}s")

        # Load audio and resample to 24k and 48k streams
        y, native_sr = librosa.load(audio_path, sr=None, mono=True)
        if y.ndim > 1:
            y = librosa.to_mono(y)

        duration = len(y) / float(native_sr)
        print(f"Loaded audio: {duration:.2f}s ({duration / 60.0:.2f} min)")

        y_24k = librosa.resample(y, orig_sr=native_sr, target_sr=self.sr_mert, res_type="soxr_vhq")
        y_48k = librosa.resample(y, orig_sr=native_sr, target_sr=self.sr_clap, res_type="soxr_vhq")

        win_24k = int(self.window_sec * self.sr_mert)
        win_48k = int(self.window_sec * self.sr_clap)

        # Generate window slices
        clips_24k: List[np.ndarray] = []
        clips_48k: List[np.ndarray] = []
        timestamps: List[float] = []

        start_sec = 0.0
        while start_sec + self.window_sec <= duration or start_sec == 0.0:
            s_24 = int(start_sec * self.sr_mert)
            e_24 = s_24 + win_24k
            c_24 = y_24k[s_24:e_24]
            if len(c_24) < win_24k:
                c_24 = np.pad(c_24, (0, win_24k - len(c_24)))

            s_48 = int(start_sec * self.sr_clap)
            e_48 = s_48 + win_48k
            c_48 = y_48k[s_48:e_48]
            if len(c_48) < win_48k:
                c_48 = np.pad(c_48, (0, win_48k - len(c_48)))

            clips_24k.append(c_24)
            clips_48k.append(c_48)
            timestamps.append(start_sec)

            start_sec += self.hop_sec
            if start_sec + self.window_sec > duration:
                break

        num_windows = len(timestamps)
        print(f"Streaming inference across {num_windows} sliding windows...")

        all_probs: List[np.ndarray] = []
        total_batches = (num_windows + self.batch_size - 1) // self.batch_size

        for i in tqdm(range(0, num_windows, self.batch_size), desc="Scoring film windows", total=total_batches):
            b_24 = clips_24k[i:i + self.batch_size]
            b_48 = clips_48k[i:i + self.batch_size]

            # Extract 1,280-d frozen embeddings
            embeddings = self.feature_extractor.extract_batch(b_24, b_48).to(self.device)

            # Evaluate with probing head
            with torch.no_grad():
                logits = self.probing_head(embeddings)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)

        P = np.vstack(all_probs)
        timestamps_arr = np.array(timestamps)
        print(f"Scoring complete: Raw probability matrix shape = {P.shape}")

        if output_probabilities_path:
            Path(output_probabilities_path).parent.mkdir(parents=True, exist_ok=True)
            np.save(output_probabilities_path, P)
            np.save(str(output_probabilities_path).replace(".npy", "_timestamps.npy"), timestamps_arr)
            print(f"Saved raw probabilities to: {output_probabilities_path}")

        return P, timestamps_arr
