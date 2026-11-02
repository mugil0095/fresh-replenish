import pytest

from fresh_replenish.analysis.drivers import build_frame
from fresh_replenish.config import DATASET_FILES, RAW_DIR
from fresh_replenish.data.db import connect

PATHS = [RAW_DIR / name for name in DATASET_FILES]

pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(
        not all(p.exists() for p in PATHS),
        reason="run `uv run python -m fresh_replenish.data.download` first",
    ),
]


def test_frame_is_train_minus_zero_discount_rows():
    con = connect(*PATHS)
    zero = con.execute(
        "SELECT count(*) FROM daily WHERE split = 'train' AND discount = 0"
    ).fetchall()[0][0]
    assert len(build_frame(con)) + zero == 4_500_000
