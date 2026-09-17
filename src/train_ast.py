"""
Train AST Probing Head Pipeline
Trains the 2-layer MLP Probing Head (1,280 -> 512 -> 256 -> 12)
using 5-fold GroupKFold by original recording file on frozen MERT+CLAP embeddings.

Usage:
    PYTHONPATH=src venv/bin/python src/train_ast.py
    PYTHONPATH=src venv/bin/python src/train_ast.py --full-prep  # Runs standardization & feature extraction first
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from soundtrack_classifier.ast.pipeline import OperationalPipeline
from soundtrack_classifier.ast.probing_head import ProbingHeadTrainer


def main():
    parser = argparse.ArgumentParser(description="Train AST Probing Head on Frozen MERT+CLAP Embeddings")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--features-dir", type=str, default=None, help="Directory containing X.pt, Y.pt, groups.pt")
    parser.add_argument("--checkpoint", type=str, default=None, help="Output checkpoint file path")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--full-prep", action="store_true", help="Run Phase 1 (Standardize), Phase 2 (Augment), Phase 3 (Feature Extraction) before training")
    args = parser.parse_args()

    pipeline = OperationalPipeline(config_path=args.config)
    ast_cfg = pipeline.ast_cfg
    paths_cfg = ast_cfg.get("paths", {})
    head_cfg = ast_cfg.get("probing_head", {})

    feat_dir = Path(args.features_dir or paths_cfg.get("features_dir", "data/ast/features"))
    x_path = feat_dir / "X.pt"
    y_path = feat_dir / "Y.pt"
    g_path = feat_dir / "groups.pt"

    if args.full_prep or not (x_path.exists() and y_path.exists()):
        print("Running full preparation: Phases 1, 2, and 3...")
        pipeline.phase_1_standardize()
        pipeline.phase_2_slice_and_augment()
        pipeline.phase_3_extract_features()

    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Feature matrices not found in {feat_dir}. Run with --full-prep to generate them.")

    print(f"Loading precomputed feature matrices from {feat_dir}...")
    X = torch.load(str(x_path), weights_only=False).numpy()
    Y = torch.load(str(y_path), weights_only=False).numpy()
    groups = torch.load(str(g_path), weights_only=False) if g_path.exists() else np.arange(len(X))

    ckpt_path = args.checkpoint or paths_cfg.get("model_checkpoint", "models/ast_probing_head.pt")

    trainer = ProbingHeadTrainer(
        in_dim=head_cfg.get("in_dim", 1280),
        hidden_dims=head_cfg.get("hidden_dims", [512, 256]),
        num_classes=head_cfg.get("num_classes", 12),
        dropouts=head_cfg.get("dropouts", [0.4, 0.3]),
        lr=float(args.lr or head_cfg.get("lr", 0.001)),
        weight_decay=float(head_cfg.get("weight_decay", 0.01)),
        epochs=int(args.epochs or head_cfg.get("epochs", 40)),
        batch_size=int(args.batch_size or head_cfg.get("batch_size", 64)),
        early_stopping_patience=int(head_cfg.get("early_stopping_patience", 10)),
        n_splits=int(head_cfg.get("n_splits", 5))
    )

    model, metrics = trainer.train_kfold(X, Y, groups=groups, save_checkpoint_path=ckpt_path)
    print("Training finished successfully!")


if __name__ == "__main__":
    main()
