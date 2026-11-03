"""Download FreshRetailNet-50K at a pinned revision and check each file's size."""

import sys
from collections.abc import Callable
from pathlib import Path

from huggingface_hub import hf_hub_download

from fresh_replenish.config import DATASET_FILES, DATASET_REPO, DATASET_REVISION, RAW_DIR


class DatasetIntegrityError(ValueError):
    """A downloaded file does not have the expected size."""


def download_dataset(
    dest: Path = RAW_DIR,
    files: dict[str, int] = DATASET_FILES,
    fetch: Callable[..., str] = hf_hub_download,
) -> dict[str, Path]:
    """Return {"train": path, "eval": path}, downloading only files that are missing or wrong."""
    paths = {}
    for name, size in files.items():
        path = Path(dest) / name
        if not (path.exists() and path.stat().st_size == size):
            path.unlink(missing_ok=True)  # drop a short file left by an interrupted download
            fetch(
                repo_id=DATASET_REPO,
                filename=name,
                repo_type="dataset",
                revision=DATASET_REVISION,
                local_dir=dest,
            )
            actual = path.stat().st_size if path.exists() else 0
            if actual != size:
                raise DatasetIntegrityError(f"{path}: expected {size:,} bytes, got {actual:,}")
        paths[Path(name).stem] = path
    return paths


def local_paths(raw_dir: Path = RAW_DIR) -> list[Path] | None:
    """The downloaded files (train, eval), or None after printing how to download them."""
    paths = [Path(raw_dir) / name for name in DATASET_FILES]
    if all(p.exists() for p in paths):
        return paths
    print("Data not found. Run: uv run python -m fresh_replenish.data.download", file=sys.stderr)
    return None


if __name__ == "__main__":
    for split, path in download_dataset().items():
        print(f"{split}: {path} ({path.stat().st_size:,} bytes)")
