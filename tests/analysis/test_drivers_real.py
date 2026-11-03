import pytest

from fresh_replenish.analysis.drivers import build_frame
from fresh_replenish.data.db import connect

pytestmark = pytest.mark.data


def test_frame_is_train_minus_zero_discount_rows(real_paths):
    con = connect(*real_paths)
    zero = con.execute(
        "SELECT count(*) FROM daily WHERE split = 'train' AND discount = 0"
    ).fetchall()[0][0]
    assert len(build_frame(con)) + zero == 4_500_000
