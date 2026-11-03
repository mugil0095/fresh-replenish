import numpy as np
import pytest

from fresh_replenish.data.db import connect
from fresh_replenish.imputation.data import load_hours


def hours(paths, **kwargs):
    return load_hours(connect(*paths), **kwargs)


def test_shapes(tiny_dataset):
    h = hours(tiny_dataset)
    assert h.sales.shape == (4, 3, 16)
    assert h.covariates.shape == (4, 3, 4)
    assert h.night.shape == h.sale_amount.shape == h.stockout_hours.shape == (4, 3)
    assert len(h.ids) == 4
    assert list(h.dates) == ["2024-01-01", "2024-01-02", "2024-01-03"]


def test_stockout_hours_are_blank(tiny_dataset):
    h = hours(tiny_dataset)
    assert list(np.flatnonzero(np.isnan(h.sales[0, 0]))) == [14, 15]  # hours 20 and 21
    assert np.isnan(h.sales).sum() == 2


def test_night_is_rest_of_day(make_dataset):
    def night_sale(train, eval_):
        train[1]["hours_sale"][2] = 0.3  # 2:00, outside the operating window
        train[1]["sale_amount"] += 0.3

    h = hours(make_dataset(night_sale))
    assert h.night[0, 1] == pytest.approx(0.3)
    assert h.night[0, 0] == pytest.approx(0.0)


def test_store_filter(tiny_dataset):
    h = hours(tiny_dataset, stores=[2])
    assert len(h.ids) == 2
    assert set(h.ids.store_id) == {2}


def test_uneven_days_raise(make_dataset):
    with pytest.raises(ValueError):
        hours(make_dataset(lambda train, eval_: train.pop(5)))
