"""Where the engine reads its inputs and writes its outputs, all inside the repository."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
AURIGA_ROOT = REPO_ROOT / "staging" / "auriga"
DATA_DIR = REPO_ROOT / "assets" / "data"
FIGURES_DIR = REPO_ROOT / "assets" / "figures"
