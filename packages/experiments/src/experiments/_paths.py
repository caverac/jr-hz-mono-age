"""Where the engine reads its inputs and writes its outputs, all inside the repository.

``staging/`` is ignored by git: it holds the Auriga snapshots and the mock snapshot. The
data commands write their small results to ``assets/data/`` and the plot commands read
them from there and write figures to ``assets/figures/``, which the paper and the
documentation site share.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
AURIGA_ROOT = REPO_ROOT / "staging" / "auriga"
MOCK_ROOT = REPO_ROOT / "staging" / "mock"
DATA_DIR = REPO_ROOT / "assets" / "data"
AURIGA_DATA_DIR = DATA_DIR / "auriga"
FIGURES_DIR = REPO_ROOT / "assets" / "figures"
