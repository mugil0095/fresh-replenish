from pathlib import Path

import pytest

from fresh_replenish.config import DATASET_FILES, DATASET_REPO, DATASET_REVISION
from fresh_replenish.data.download import DatasetIntegrityError, download_dataset, local_paths

SIZES = {"data/train.parquet": 10, "data/eval.parquet": 5}


class FakeFetch:
    """Stands in for hf_hub_download: writes `sizes[filename]` bytes and records each call."""

    def __init__(self, sizes):
        self.sizes = sizes
        self.calls = []

    def __call__(self, *, repo_id, filename, repo_type, revision, local_dir):
        self.calls.append(
            {"repo_id": repo_id, "filename": filename, "repo_type": repo_type, "revision": revision}
        )
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * self.sizes[filename])
        return str(path)


def _write(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


def test_returns_both_splits(tmp_path):
    paths = download_dataset(tmp_path, SIZES, fetch=FakeFetch(SIZES))
    assert set(paths) == {"train", "eval"}
    assert all(p.exists() for p in paths.values())


def test_passes_pinned_revision(tmp_path):
    fetch = FakeFetch(SIZES)
    download_dataset(tmp_path, SIZES, fetch=fetch)
    assert len(fetch.calls) == 2
    for call in fetch.calls:
        assert call["revision"] == DATASET_REVISION
        assert call["repo_type"] == "dataset"
        assert call["repo_id"] == DATASET_REPO


def test_skips_file_with_right_size(tmp_path):
    _write(tmp_path / "data/train.parquet", 10)
    fetch = FakeFetch(SIZES)
    download_dataset(tmp_path, SIZES, fetch=fetch)
    assert [c["filename"] for c in fetch.calls] == ["data/eval.parquet"]


def test_refetches_short_file(tmp_path):
    _write(tmp_path / "data/train.parquet", 3)  # left behind by an interrupted download
    fetch = FakeFetch(SIZES)
    paths = download_dataset(tmp_path, SIZES, fetch=fetch)
    assert "data/train.parquet" in [c["filename"] for c in fetch.calls]
    assert paths["train"].stat().st_size == 10


def test_raises_when_size_still_wrong(tmp_path):
    wrong = FakeFetch({"data/train.parquet": 7, "data/eval.parquet": 5})
    with pytest.raises(DatasetIntegrityError):
        download_dataset(tmp_path, SIZES, fetch=wrong)


def test_local_paths_when_downloaded(tmp_path):
    for name in DATASET_FILES:
        _write(tmp_path / name, 1)
    assert local_paths(tmp_path) == [tmp_path / name for name in DATASET_FILES]


def test_local_paths_hints_download_when_missing(tmp_path, capsys):
    _write(tmp_path / "data/train.parquet", 1)  # eval is missing
    assert local_paths(tmp_path) is None
    assert "fresh_replenish.data.download" in capsys.readouterr().err
