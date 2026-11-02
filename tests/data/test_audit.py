import shutil

import pytest

from fresh_replenish.data.audit import main, render_report, run_audit
from fresh_replenish.data.db import connect

TINY_ROWS = {"train": 12, "eval": 4}


def audit(paths):
    return run_audit(connect(*paths), TINY_ROWS)


def failed(result) -> set[str]:
    return {c.name for c in result.checks if not c.passed}


def test_clean_fixture_passes(tiny_dataset):
    result = audit(tiny_dataset)
    assert result.passed, failed(result)


def test_duplicate_key_fails(make_dataset):
    result = audit(make_dataset(lambda train, eval_: train.append(dict(train[0]))))
    assert "duplicate_keys" in failed(result)


def test_bad_list_length_fails(make_dataset):
    def cut(train, eval_):
        train[1]["hours_sale"] = train[1]["hours_sale"][:23]

    assert "bad_hour_lists" in failed(audit(make_dataset(cut)))


def test_stock_count_mismatch_fails(make_dataset):
    def wrong_count(train, eval_):
        train[1]["stock_hour6_22_cnt"] = 5

    assert "stock_count_mismatch" in failed(audit(make_dataset(wrong_count)))


def test_stockout_by_hour_values(tiny_dataset):
    table = audit(tiny_dataset).tables["stockout_by_hour"].set_index("hour")["stockout_share"]
    assert list(table.index) == list(range(6, 22))
    assert table[20] == pytest.approx(1 / 16)  # one of 16 rows is out of stock at 20:00 and 21:00
    assert table[21] == pytest.approx(1 / 16)
    assert table.drop([20, 21]).eq(0).all()


def test_report_has_headline(tiny_dataset):
    assert "Stockout share of operating hours" in render_report(audit(tiny_dataset))


def test_cli_missing_data_message(tmp_path, capsys):
    assert main(raw_dir=tmp_path / "empty", out_dir=tmp_path / "out") == 2
    assert "fresh_replenish.data.download" in capsys.readouterr().err


def test_cli_writes_report_even_when_checks_fail(tiny_dataset, tmp_path):
    raw = tmp_path / "raw" / "data"
    raw.mkdir(parents=True)
    for path in tiny_dataset:
        shutil.copy(path, raw / path.name)
    out = tmp_path / "out"
    assert main(raw_dir=tmp_path / "raw", out_dir=out) == 1  # tiny row counts != real ones
    assert "FAILED" in (out / "data_audit.md").read_text(encoding="utf-8")
    assert (out / "tables" / "stockout_by_hour.csv").exists()
