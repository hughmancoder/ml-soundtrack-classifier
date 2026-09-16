#!/usr/bin/env python3
"""
Backward-compatibility wrapper.
Film audio slicing has been generalized into scripts/data/slice_film.py.
"""
import sys
from pathlib import Path

# Ensure src/ is in sys.path
repo_root = Path(__file__).resolve().parents[2]
src_dir = str(repo_root / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from scripts.data.slice_film import main

if __name__ == "__main__":
    print("[Notice] slice_nonmusic.py has been updated to slice_film.py (supporting multi-class labels). Delegating...")
    main()
