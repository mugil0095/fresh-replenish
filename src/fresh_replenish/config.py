"""Pinned dataset facts shared across the project."""

from pathlib import Path

DATASET_REPO = "Dingdong-Inc/FreshRetailNet-50K"
DATASET_REVISION = "c374e85975c82816c495acfa904cc55ab14a47d3"
DATASET_FILES: dict[str, int] = {
    "data/train.parquet": 106_436_287,
    "data/eval.parquet": 8_440_124,
}
RAW_DIR = Path("data/raw")
