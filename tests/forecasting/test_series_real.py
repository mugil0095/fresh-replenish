import numpy as np
import pytest

from fresh_replenish.config import PROCESSED_DIR
from fresh_replenish.data.db import connect, sample_stores
from fresh_replenish.forecasting.data import load_series

pytestmark = pytest.mark.data


def test_store_sample_has_phase_3_series_and_90_days(real_paths):
    recovered = PROCESSED_DIR / "recovered_demand.parquet"
    if not recovered.exists():
        pytest.skip("run `uv run python -m fresh_replenish.imputation.recovery` first")
    con = connect(*real_paths)
    s = load_series(con, recovered, stores=sample_stores(con, 0.10, np.random.default_rng(42)))
    assert s.demand.shape == (4928, 90)
    assert (s.dates[0], s.dates[-1]) == ("2024-03-28", "2024-06-25")
    assert np.isfinite(s.demand).all() and (s.demand >= 0).all()
