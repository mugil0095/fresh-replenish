import numpy as np
import pytest

from fresh_replenish.forecasting.data import ATTRIBUTES, HORIZON, ORIGINS


def test_shapes(tiny_dataset, series_from):
    s = series_from(tiny_dataset)
    assert s.demand.shape == s.in_stock.shape == (4, 3)
    assert s.inputs.shape == (4, 3, 3)
    assert list(s.ids.columns) == ["store_id", "product_id", *ATTRIBUTES]
    assert list(s.dates) == ["2024-01-01", "2024-01-02", "2024-01-03"]


def test_demand_is_recovered_and_stockouts_flagged(tiny_dataset, series_from):
    s = series_from(tiny_dataset)
    assert s.demand[0, 0] == pytest.approx(1.2)  # 0.2 sold + 1.0 recovered
    assert not s.in_stock[0, 0]
    assert s.in_stock.sum() == 11


def test_inputs_in_order(make_dataset, series_from):
    def plans(train, eval_):
        train[1].update(discount=0.8, activity_flag=1)
        train[2]["holiday_flag"] = 1

    s = series_from(make_dataset(plans))
    assert s.inputs[0, 1].tolist() == pytest.approx([0.8, 1.0, 0.0])  # discount, promo, holiday
    assert s.inputs[0, 2].tolist() == pytest.approx([1.0, 0.0, 1.0])


def test_ids_carry_product_attributes(tiny_dataset, series_from):
    s = series_from(tiny_dataset)
    assert s.ids.first_category_id.tolist() == [5, 6, 5, 6]
    assert s.ids.city_id.tolist() == [0, 0, 1, 1]


def test_store_filter(tiny_dataset, series_from):
    s = series_from(tiny_dataset, stores=[2])
    assert set(s.ids.store_id) == {2}
    assert s.demand.shape == (2, 3)


def test_missing_day_raises(make_dataset, series_from):
    with pytest.raises(ValueError, match="same number"):
        series_from(make_dataset(lambda train, eval_: train.pop(4)))


def test_folds_end_on_last_train_day():
    assert ORIGINS[-1] + HORIZON == 90
    assert np.diff(ORIGINS).tolist() == [HORIZON] * 3


def test_unknown_store_says_no_rows(tiny_dataset, series_from):
    with pytest.raises(ValueError, match="no rows"):
        series_from(tiny_dataset, stores=[99])


def test_eval_days_come_from_a_second_file(tiny_dataset, series_from):
    s = series_from(tiny_dataset, with_eval=True, splits=("train", "eval"))
    assert s.demand.shape == (4, 4)
    assert s.inputs.shape == (4, 4, 3)
    assert s.dates[-1] == "2024-01-04"


def test_eval_rows_stay_out_unless_asked_for(tiny_dataset, series_from):
    assert series_from(tiny_dataset, with_eval=True).demand.shape == (4, 3)
