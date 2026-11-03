import numpy as np
import pytest

from fresh_replenish.metrics import coverage, mase, wql


def test_wql_on_hand_example():
    actual = np.array([[10.0]])
    assert wql(np.full((1, 9, 1), 8.0), actual) == pytest.approx(0.2)  # 2 * mean(tau * 2) / 10
    assert wql(np.full((1, 9, 1), 10.0), actual) == 0.0


def test_coverage_counts_actuals_inside_the_outer_deciles():
    q = np.zeros((1, 9, 4))
    q[:, 0], q[:, -1] = 1.0, 3.0
    assert coverage(q, np.array([[0.0, 2.0, 3.0, 4.0]])) == 0.5
    assert np.isnan(coverage(np.full((1, 9, 4), np.nan), np.ones((1, 4))))


def test_mase_scales_by_seasonal_naive_and_skips_flat_history():
    history = np.array([[1, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7, 8], [3.0] * 14])
    ratios = mase(np.array([[5.0], [9.0]]), np.array([[3.0], [1.0]]), history)
    assert ratios.tolist() == [2.0]  # MAE 2 over a seasonal-naive MAE of 1; series 2 is flat
