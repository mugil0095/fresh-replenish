import pytest
import streamlit as st
from fastapi.testclient import TestClient
from smoke_data import COUNT, write
from streamlit.testing.v1 import AppTest

from fresh_replenish import config
from fresh_replenish.config import PROCESSED_DIR, REPO_ROOT
from fresh_replenish.serving.api import create_app

APP = REPO_ROOT / "src" / "fresh_replenish" / "dashboard" / "app.py"


def _opens_with_the_headline(app):
    assert not app.exception and not app.error
    assert "Cost-based orders cut" in app.markdown[0].value
    assert len(app.metric) == 4 and len(app.tabs) == 5
    assert [s.value for s in app.subheader] == ["A second delivery"]


@pytest.mark.data
def test_dashboard_opens_with_the_headline(real_paths):
    if not (PROCESSED_DIR / "forecast_eval.parquet").exists():
        pytest.skip("run the pipeline first (see the README)")
    st.cache_data.clear()
    _opens_with_the_headline(AppTest.from_file(str(APP), default_timeout=300).run())


def test_dashboard_opens_on_the_smoke_data(tmp_path, monkeypatch):
    write(tmp_path)
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "raw")
    st.cache_data.clear()  # load() is cached by its code, not by these paths
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    _opens_with_the_headline(app)
    assert "1 store-product series in 1 dark stores" in app.caption[0].value


def test_the_smoke_data_never_replaces_real_outputs(tmp_path):
    write(tmp_path)
    with pytest.raises(FileExistsError):
        write(tmp_path)


def test_the_api_serves_the_smoke_orders(tmp_path):
    write(tmp_path)
    client = TestClient(create_app(tmp_path / "processed" / "orders.parquet"))
    assert client.get("/reorder", params={"store_id": 1}).status_code == 200
    assert client.post("/topup", json=COUNT).status_code == 200
