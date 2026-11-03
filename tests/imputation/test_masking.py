import numpy as np

from fresh_replenish.imputation.masking import mimic_missing


def windows(stockout_share: float = 0.0, seed: int = 0) -> np.ndarray:
    """50 windows x 30 days x 16 hours of sales, with some real stockouts blanked."""
    rng = np.random.default_rng(seed)
    x = rng.random((50, 30, 16)).astype(np.float32)
    x[rng.random(x.shape) < stockout_share] = np.nan
    return x


def test_only_fully_in_stock_days_get_hidden_hours():
    x = windows(stockout_share=0.02)
    _, valid = mimic_missing(x, np.random.default_rng(1))
    assert not valid[np.isnan(x).any(axis=-1)].any()


def test_real_blanks_stay_and_valid_marks_only_new_blanks():
    x = windows(stockout_share=0.02)
    masked, valid = mimic_missing(x, np.random.default_rng(1))
    assert np.isnan(masked[np.isnan(x)]).all()
    assert (valid == (~np.isnan(x) & np.isnan(masked))).all()
    assert valid.any()


def test_same_seed_same_mask():
    x = windows()
    a = mimic_missing(x, np.random.default_rng(7))[1]
    b = mimic_missing(x, np.random.default_rng(7))[1]
    assert (a == b).all()


def test_hidden_share_is_plausible_at_p_0_3():
    share = mimic_missing(windows(), np.random.default_rng(3))[1].mean()
    assert 0.05 < share < 0.6
