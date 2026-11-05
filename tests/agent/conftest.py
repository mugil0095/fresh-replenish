import numpy as np
import pandas as pd
import pytest

from fresh_replenish.agent.tools import Toolbox
from fresh_replenish.metrics import DECILE_COLUMNS, REST_COLUMNS


@pytest.fixture
def orders():
    """Batch orders for 2 stores x 2 products x 2 days: deciles 1 ... 9, ordered at q90 (9).

    Rest-of-day deciles run 0.5 ... 4.5, for a shelf count at 13:00 and a delivery at 15:00.
    """
    rows = pd.DataFrame(
        [(s, p, f"2024-06-2{6 + d}") for s in (1, 2) for p in (10, 11) for d in (0, 1)],
        columns=["store_id", "product_id", "dt"],
    )
    rows[DECILE_COLUMNS] = np.linspace(1, 9, 9)
    rows[REST_COLUMNS] = np.linspace(0.5, 4.5, 9)
    return rows.assign(order=9.0, as_of="2024-06-25", ratio=9.0, decision_hour=13, arrival_hour=15)


@pytest.fixture
def toolbox(orders, tmp_path):
    def scenario(store_id, product_id, date, changes):
        return None if product_id == 99 else (2.0, 2.5)

    return Toolbox(orders, scenario=scenario, records=tmp_path / "records.jsonl")
