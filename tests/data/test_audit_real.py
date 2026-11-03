import pytest

from fresh_replenish.data.audit import run_audit
from fresh_replenish.data.db import connect

pytestmark = pytest.mark.data


def test_real_data_passes_hard_checks(real_paths):
    result = run_audit(connect(*real_paths))
    assert result.passed, [c for c in result.checks if not c.passed]
    overview = result.tables["overview"].set_index("split")
    assert overview.loc["train", "series"] == 50_000
    assert overview.loc["train", "stores"] == 898
    assert overview.loc["train", "cities"] == 18
    assert overview.loc["train", "days"] == 90
    assert overview.loc["eval", "days"] == 7
