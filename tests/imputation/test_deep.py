import numpy as np
import pytest

from fresh_replenish.imputation.deep import deep_recover


@pytest.mark.parametrize("model", ["saits", "timesnet"])
def test_fills_blanks_and_keeps_observed(model):
    rng = np.random.default_rng(0)
    sales = rng.random((6, 6, 16)).astype(np.float32)
    sales[rng.random(sales.shape) < 0.2] = np.nan
    covariates = rng.random((6, 6, 4)).astype(np.float32)
    out = deep_recover(sales, covariates, model, epochs=1, device="cpu")
    observed = ~np.isnan(sales)
    assert out.shape == sales.shape
    assert not np.isnan(out).any()
    assert np.array_equal(out[observed], sales[observed])


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        deep_recover(np.zeros((3, 3, 16), np.float32), np.zeros((3, 3, 4), np.float32), "lstm")
