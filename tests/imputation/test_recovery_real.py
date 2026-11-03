import numpy as np
import pytest

from fresh_replenish.data.db import connect
from fresh_replenish.imputation.data import load_hours

pytestmark = pytest.mark.data


def test_one_store_has_90_days_and_blanks_match_stockouts(real_paths):
    con = connect(*real_paths)
    h = load_hours(con, stores=[0])
    assert h.sales.shape[1:] == (90, 16)
    share = con.execute(
        "SELECT sum(stockout_hours) / (16.0 * count(*)) FROM daily"
        " WHERE split = 'train' AND store_id = 0"
    ).fetchall()[0][0]
    assert np.isnan(h.sales).mean() == pytest.approx(share)
