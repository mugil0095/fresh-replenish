import pytest

from fresh_replenish.data.db import connect

FIRST_ROW = "store_id = 1 AND product_id = 10 AND dt = '2024-01-01'"


def test_views_build_from_path_with_spaces(tiny_dataset):
    train_path, eval_path = tiny_dataset
    assert " " in str(train_path.parent)
    con = connect(train_path, eval_path)
    assert con.execute("SELECT count(*) FROM train").fetchall()[0][0] == 12
    assert con.execute("SELECT count(*) FROM eval").fetchall()[0][0] == 4


def test_daily_has_split_column(tiny_dataset):
    con = connect(*tiny_dataset)
    counts = dict(con.execute("SELECT split, count(*) FROM daily GROUP BY split").fetchall())
    assert counts == {"train": 12, "eval": 4}


def test_stockout_hours_use_operating_window(tiny_dataset):
    con = connect(*tiny_dataset)
    hours = con.execute(f"SELECT stockout_hours FROM daily WHERE {FIRST_ROW}").fetchall()[0][0]
    assert hours == 2  # night-hour flags (0-2, 22-23) are outside hours 6-21


def test_hours_sale_sum_matches_sale_amount(tiny_dataset):
    con = connect(*tiny_dataset)
    total = con.execute(f"SELECT hours_sale_sum FROM daily WHERE {FIRST_ROW}").fetchall()[0][0]
    assert total == pytest.approx(0.2)
