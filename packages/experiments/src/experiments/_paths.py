"""Where the engine reads its inputs and writes its outputs, all inside the repository.

``staging/`` is ignored by git: it holds the Auriga snapshots, the mock snapshot and the
results of the analysis, which stay local until the paper is public.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
AURIGA_ROOT = REPO_ROOT / "staging" / "auriga"
MOCK_ROOT = REPO_ROOT / "staging" / "mock"
RESULTS_DIR = REPO_ROOT / "staging" / "results"
DATA_DIR = REPO_ROOT / "assets" / "data"
FIGURES_DIR = REPO_ROOT / "assets" / "figures"
