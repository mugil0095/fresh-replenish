import pytest
from streamlit.testing.v1 import AppTest

from fresh_replenish.config import PROCESSED_DIR, REPO_ROOT

pytestmark = pytest.mark.data

APP = REPO_ROOT / "src" / "fresh_replenish" / "dashboard" / "app.py"


def test_dashboard_opens_with_the_headline(real_paths):
    if not (PROCESSED_DIR / "forecast_eval.parquet").exists():
        pytest.skip("run the pipeline first (see the README)")
    app = AppTest.from_file(str(APP), default_timeout=300).run()
    assert not app.exception
    assert "Cost-based orders cut" in app.markdown[0].value
    assert len(app.metric) == 4 and len(app.tabs) == 5
    assert [s.value for s in app.subheader] == ["A second delivery"]
