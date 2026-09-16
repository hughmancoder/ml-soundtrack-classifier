import os
import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoProcessor, ClapModel
from tqdm import tqdm


class FrozenFeatureExtractor:
    """
    Phase 3: Frozen Foundation Feature Extraction
    Loads frozen MERT-v1-95M (768-d) and LAION-CLAP (512-d).
    - Extracts 768-d acoustic representation from 24kHz audio
    - Extracts 512-d audio projection from 48kHz audio
    - Applies L2 feature normalization to each stream
    - Concatenates into single 1,280-d embedding vector
    - Saves precomputed feature matrix X (N, 1280) and label matrix Y (N, 12)
    """
    def __init__(
        self,
        mert_model_id: str = "m-a-p/MERT-v1-95M",
        clap_model_id: str = "laion/clap-htsat-unfused",
        device: Optional[str] = None,
        fp16: bool = True,
        batch_size: int = 16
    ):
        self.mert_model_id = mert_model_id
        self.clap_model_id = clap_model_id
        self.fp16 = fp16
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

        self.mert_model: Optional[AutoModel] = None
        self.clap_model: Optional[ClapModel] = None
        self.clap_processor: Optional[AutoProcessor] = None

    def load_models(self):
        """
        Loads frozen models onto designated hardware (MPS/CPU) in FP16 / FP32.
        """
        if self.mert_model is not None and self.clap_model is not None:
            return

        print(f"Loading frozen MERT ({self.mert_model_id}) on {self.device}...")
        self.mert_model = AutoModel.from_pretrained(
            self.mert_model_id,
            trust_remote_code=True
        )
        self.mert_model.eval()
        for param in self.mert_model.parameters():
            param.requires_grad = False

        print(f"Loading frozen LAION-CLAP ({self.clap_model_id}) on {self.device}...")
        self.clap_model = ClapModel.from_pretrained(self.clap_model_id)
        self.clap_processor = AutoProcessor.from_pretrained(self.clap_model_id)
        self.clap_model.eval()
        for param in self.clap_model.parameters():
            param.requires_grad = False

        # Move to device
        self.mert_model.to(self.device)
        self.clap_model.to(self.device)

        # Apply FP16 on GPU where appropriate
        if self.fp16 and self.device.type in ("mps", "cuda"):
            self.mert_model = self.mert_model.half()
            self.clap_model = self.clap_model.half()

        print("Foundation models successfully loaded and frozen.")

    def extract_batch(
        self,
        audio_24k_list: List[np.ndarray],
        audio_48k_list: List[np.ndarray]
    ) -> torch.Tensor:
        """
        Extracts L2-normalized, fused 1,280-d embeddings for a batch of audio clips.
        Args:
            audio_24k_list: List of 1D numpy arrays (24kHz)
            audio_48k_list: List of 1D numpy arrays (48kHz)
        Returns:
            fused_embeddings: Tensor of shape [Batch, 1280]
        """
        self.load_models()

        # 1. MERT Forward Pass (24 kHz)
        t_24 = torch.tensor(np.stack(audio_24k_list), dtype=torch.float32).to(self.device)
        if self.fp16 and self.device.type in ("mps", "cuda"):
            t_24 = t_24.half()

        with torch.no_grad():
            mert_out = self.mert_model(t_24)
            # Pool last hidden state across time frames
            if hasattr(mert_out, "last_hidden_state"):
                mert_emb = mert_out.last_hidden_state.mean(dim=1)  # [B, 768]
            elif hasattr(mert_out, "pooler_output") and mert_out.pooler_output is not None:
                mert_emb = mert_out.pooler_output
            else:
                mert_emb = mert_out[0].mean(dim=1)

            # 2. CLAP Forward Pass (48 kHz)
            clap_inputs = self.clap_processor(
                audio=audio_48k_list,
                sampling_rate=48000,
                return_tensors="pt"
            )
            clap_inputs = {k: v.to(self.device) for k, v in clap_inputs.items()}
            if self.fp16 and self.device.type in ("mps", "cuda"):
                # Cast floating point inputs to half
                for k in clap_inputs:
                    if clap_inputs[k].dtype == torch.float32:
                        clap_inputs[k] = clap_inputs[k].half()

            clap_out = self.clap_model.get_audio_features(**clap_inputs)
            if hasattr(clap_out, "pooler_output") and clap_out.pooler_output is not None:
                clap_emb = clap_out.pooler_output  # [B, 512]
            elif isinstance(clap_out, torch.Tensor):
                clap_emb = clap_out
            else:
                clap_emb = clap_out[0]

            # 3. L2 Feature Normalization
            mert_norm = F.normalize(mert_emb.float(), p=2, dim=-1, eps=1e-8)
            clap_norm = F.normalize(clap_emb.float(), p=2, dim=-1, eps=1e-8)

            # 4. Concatenate into single 1,280-d embedding
            fused = torch.cat([mert_norm, clap_norm], dim=-1)  # [B, 1280]

        return fused.cpu()

    def extract_from_manifest(
        self,
        manifest_path: str = "data/ast/augmented_segments/segments_manifest.pt",
        output_features_dir: str = "data/ast/features"
    ) -> Tuple[str, str, str]:
        """
        Processes all segment tensors from the manifest and saves pre-computed
        feature matrix X, label matrix Y, and group recording IDs to disk.
        Supports both memory-efficient sharded manifests and single-tensor manifests.
        """
        print(f"Phase 3: Loading augmented segments from {manifest_path}...")
        data = torch.load(manifest_path, weights_only=False)

        X_list: List[torch.Tensor] = []
        Y_list: List[np.ndarray] = []
        groups_list: List[str] = []

        if isinstance(data, dict) and "shards" in data:
            shard_paths = data["shards"]
            total_segs = data.get("total_segments", len(shard_paths))
            print(f"Loaded sharded manifest with {len(shard_paths)} shards ({total_segs} segments total).")

            for shard_file in tqdm(shard_paths, desc="Extracting features across shards"):
                shard_segs = torch.load(shard_file, weights_only=False)
                for i in range(0, len(shard_segs), self.batch_size):
                    batch_segs = shard_segs[i:i + self.batch_size]
                    b_24 = [s["audio_24k"] for s in batch_segs]
                    b_48 = [s["audio_48k"] for s in batch_segs]

                    embeddings = self.extract_batch(b_24, b_48)
                    X_list.append(embeddings)

                    for s in batch_segs:
                        Y_list.append(s["labels"])
                        groups_list.append(s["recording_id"])
                del shard_segs
                gc.collect()
        else:
            segments: List[Dict[str, Any]] = data
            print(f"Loaded {len(segments)} segments for frozen feature extraction.")
            total_batches = (len(segments) + self.batch_size - 1) // self.batch_size

            for i in tqdm(range(0, len(segments), self.batch_size), desc="Extracting MERT+CLAP embeddings", total=total_batches):
                batch_segs = segments[i:i + self.batch_size]
                b_24 = [s["audio_24k"] for s in batch_segs]
                b_48 = [s["audio_48k"] for s in batch_segs]

                embeddings = self.extract_batch(b_24, b_48)
                X_list.append(embeddings)

                for s in batch_segs:
                    Y_list.append(s["labels"])
                    groups_list.append(s["recording_id"])

        X_tensor = torch.cat(X_list, dim=0)
        Y_tensor = torch.tensor(np.stack(Y_list), dtype=torch.float32)
        groups_array = np.array(groups_list)

        print(f"Extraction complete: Matrix X shape = {X_tensor.shape}, Label Y shape = {Y_tensor.shape}")

        out_dir = Path(output_features_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        x_path = out_dir / "X.pt"
        y_path = out_dir / "Y.pt"
        g_path = out_dir / "groups.pt"

        torch.save(X_tensor, str(x_path))
        torch.save(Y_tensor, str(y_path))
        torch.save(groups_array, str(g_path))

        print(f"Saved pre-computed embeddings to: {x_path}, {y_path}, {g_path}")
        return str(x_path), str(y_path), str(g_path)
