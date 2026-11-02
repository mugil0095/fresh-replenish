import math

from fresh_replenish.analysis.drivers import TERMS, build_frame
from fresh_replenish.data.db import connect


def frame(paths, **kwargs):
    return build_frame(connect(*paths), **kwargs)


def test_frame_is_train_only_with_terms(tiny_dataset):
    f = frame(tiny_dataset)
    assert len(f) == 12
    assert set(TERMS) | {"sale_amount", "store_id", "product_id", "city_id", "week"} <= set(f)


def test_zero_discount_rows_are_excluded(make_dataset):
    def zero(train, eval_):
        train[3]["discount"] = 0.0

    assert len(frame(make_dataset(zero))) == 11


def test_log_discount_is_log_price_ratio(make_dataset):
    def half_price(train, eval_):
        train[2]["discount"] = 0.5

    values = sorted(frame(make_dataset(half_price)).log_discount.round(6).unique())
    assert values == [round(math.log(0.5), 6), 0.0]


def test_in_stock_only_drops_stockout_days(tiny_dataset):
    assert len(frame(tiny_dataset, in_stock_only=True)) == 11  # fixture row 0 has 2 stockout hours


def test_exclude_cities(tiny_dataset):
    f = frame(tiny_dataset, exclude_cities=(0,))
    assert set(f.city_id) == {1}
    assert len(f) == 6


def test_week_number(tiny_dataset):
    assert set(frame(tiny_dataset).week) == {1}  # 2024-01-01..03 are in ISO week 1
