import numpy as np
import pytest

from fresh_replenish.decisions.newsvendor import critical_ratio, order_at, simulate


def test_critical_ratio_is_the_cost_share_of_a_lost_sale():
    assert critical_ratio(9) == pytest.approx(0.9)
    assert critical_ratio(1) == 0.5


def test_order_interpolates_between_neighbouring_deciles():
    q = np.arange(9.0)  # q10 = 0 ... q90 = 8
    assert order_at(q, 0.85) == pytest.approx(7.5)
    assert order_at(q, 0.9) == pytest.approx(8.0)
    assert order_at(q, 0.1) == pytest.approx(0.0)
    assert order_at(q, 1 / 3) == pytest.approx(7 / 3)


@pytest.mark.parametrize("ratio", [0.05, 0.95])
def test_order_outside_the_deciles_raises(ratio):
    with pytest.raises(ValueError, match="outside"):
        order_at(np.arange(9.0), ratio)


def test_simulate_charges_waste_and_lost_sales():
    cost, bias, demand = simulate(np.array([[3.0, 5.0]]), np.array([[5.0, 3.0]]), r=2)
    assert cost.tolist() == [6.0]  # 2 lost x 2 + 2 wasted x 1
    assert bias.tolist() == [0.0]
    assert demand.tolist() == [8.0]
