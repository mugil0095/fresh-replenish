"""Pinned dataset facts shared across the project."""

from pathlib import Path

DATASET_REPO = "Dingdong-Inc/FreshRetailNet-50K"
DATASET_REVISION = "c374e85975c82816c495acfa904cc55ab14a47d3"
DATASET_FILES: dict[str, int] = {
    "data/train.parquet": 106_436_287,
    "data/eval.parquet": 8_440_124,
}
# shortcut: assumes the editable install (uv sync); use a settings env var if the package ships.
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
REPORTS_DIR = REPO_ROOT / "reports"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"  # git-ignored, like the raw data
