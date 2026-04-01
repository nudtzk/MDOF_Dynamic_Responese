from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "data"
PAPERS_DIR = DATA_DIR / "papers"
RESULTS_DIR = ROOT_DIR / "results"


def ensure_project_dirs() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    PAPERS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
