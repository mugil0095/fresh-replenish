import pytest

from fresh_replenish.config import DATASET_FILES, RAW_DIR
from fresh_replenish.data.audit import run_audit
from fresh_replenish.data.db import connect

PATHS = [RAW_DIR / name for name in DATASET_FILES]

pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(
        not all(p.exists() for p in PATHS),
        reason="run `uv run python -m fresh_replenish.data.download` first",
    ),
]


def test_real_data_passes_hard_checks():
    result = run_audit(connect(*PATHS))
    assert result.passed, [c for c in result.checks if not c.passed]
    overview = result.tables["overview"].set_index("split")
    assert overview.loc["train", "series"] == 50_000
    assert overview.loc["train", "stores"] == 898
    assert overview.loc["train", "cities"] == 18
    assert overview.loc["train", "days"] == 90
    assert overview.loc["eval", "days"] == 7
